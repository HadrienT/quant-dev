"""Unit tests for the Postgres storage layer, with the driver mocked out.

The SQL these produce is exercised for real against a live database in
tests/integration; here we only check the layer's contract.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from storage import DB_COLUMNS, PostgresStore


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "Date": ["2025-02-07", "2025-02-07"],
            "Ticker": ["AAPL", "MSFT"],
            "Open": [228.53, 412.35],
            "High": [232.67, 413.83],
            "Low": [228.27, 410.40],
            "Close": [232.47, 413.29],
            "Volume": [39620300, 16316700],
        }
    )
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def test_db_columns_are_lowercase():
    """DataFrames use yfinance's capitalised names, Postgres does not."""
    assert DB_COLUMNS == ["date", "ticker", "open", "high", "low", "close", "volume"]


def test_upsert_empty_dataframe_touches_nothing():
    store = PostgresStore(connection_string="dbname=unused")
    with patch("storage.psycopg.connect") as connect:
        assert store.upsert(pd.DataFrame()) == 0
    connect.assert_not_called()


def test_upsert_rejects_missing_columns(sample_data):
    store = PostgresStore(connection_string="dbname=unused")
    with patch("storage.psycopg.connect"):
        with pytest.raises(ValueError, match="missing required columns"):
            store.upsert(sample_data.drop(columns=["Volume"]))


def test_upsert_stages_then_merges(sample_data):
    """Rows go through a staging table so a failure cannot half-update the main one."""
    store = PostgresStore(connection_string="dbname=unused", table="sp500_data")

    cursor = MagicMock()
    copy = MagicMock()
    cursor.copy.return_value.__enter__.return_value = copy
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch("storage.psycopg.connect") as connect:
        connect.return_value.__enter__.return_value = conn
        assert store.upsert(sample_data) == 2

    # Render the composed SQL rather than reading its repr, so the assertions
    # below are about the statement Postgres receives.
    statements = " ".join(call.args[0].as_string(None) for call in cursor.execute.call_args_list)
    assert "CREATE TEMP TABLE staging" in statements
    assert "ON COMMIT DROP" in statements
    assert "ON CONFLICT (date, ticker) DO UPDATE" in statements

    # The key columns must not be in the update list: rewriting them is a no-op
    # at best, and hides a bad join at worst.
    assert '"date" = EXCLUDED."date"' not in statements
    assert '"close" = EXCLUDED."close"' in statements

    assert copy.write_row.call_count == len(sample_data)


def test_row_count_reads_scalar():
    store = PostgresStore(connection_string="dbname=unused")
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (1234,)

    with patch("storage.psycopg.connect") as connect:
        connect.return_value.__enter__.return_value = conn
        assert store.row_count() == 1234
