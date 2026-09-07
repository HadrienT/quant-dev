"""Integration tests against a real Postgres.

Replaces the old BigQuery integration and e2e suites, which were near-identical
copies of each other pointed at the `financial_data_test` dataset. They run
against whatever `PGHOST`/`PGPORT`/... point at, and skip cleanly when there is
no database to talk to, so the unit suite stays runnable anywhere.

Locally:  docker compose up -d postgres && pytest tests/integration
"""

import datetime

import pandas as pd
import psycopg
import pytest

from config import dsn
from market_data import download_previous_day_data, get_sp500_tickers
from storage import PostgresStore

pytestmark = pytest.mark.postgres

TEST_TABLE = "sp500_data_test"


@pytest.fixture(scope="module")
def store():
    try:
        with psycopg.connect(dsn(), connect_timeout=3):
            pass
    except Exception as exc:
        pytest.skip(f"No Postgres reachable at {dsn()}: {exc}")

    store = PostgresStore(table=TEST_TABLE)
    store.ensure_schema()
    yield store

    with store.connect() as conn:
        conn.execute(f'DROP TABLE IF EXISTS "{TEST_TABLE}"')


@pytest.fixture(autouse=True)
def clean_table(store):
    with store.connect() as conn:
        conn.execute(f'TRUNCATE "{TEST_TABLE}"')
    yield


@pytest.fixture
def sample_row():
    return pd.DataFrame(
        {
            "Date": [datetime.date(2025, 2, 7)],
            "Ticker": ["AAPL"],
            "Open": [228.53],
            "High": [232.67],
            "Low": [228.27],
            "Close": [232.47],
            "Volume": [39620300],
        }
    )


@pytest.mark.network
def test_download_then_upsert(store):
    """The whole pipeline, from yfinance to a row count in Postgres."""
    tickers = get_sp500_tickers()[:5]
    assert tickers

    data = download_previous_day_data(tickers)
    if data.empty:
        pytest.skip("yfinance returned no data for the previous session")

    sent = store.upsert(data)
    assert sent == len(data)
    assert store.row_count() == len(data)

    first, last = store.date_range()
    assert first is not None and last is not None


def test_upsert_is_idempotent(store, sample_row):
    """Re-running the same session must not duplicate rows."""
    store.upsert(sample_row)
    store.upsert(sample_row)

    assert store.row_count() == 1


def test_upsert_updates_existing_row(store, sample_row):
    """A corrected price overwrites the old one, as the BigQuery MERGE did."""
    store.upsert(sample_row)

    corrected = sample_row.copy()
    corrected["Close"] = [999.99]
    store.upsert(corrected)

    stored = store.load_prices()
    assert store.row_count() == 1
    assert stored["Close"].iloc[0] == pytest.approx(999.99)


def test_round_trip_preserves_types(store, sample_row):
    """Edge values survive the COPY and come back with the right dtypes."""
    edge = sample_row.copy()
    edge["Open"] = [999999.99]
    edge["Low"] = [0.01]
    edge["Volume"] = [2147483647]

    store.upsert(edge)
    stored = store.load_prices()

    assert stored["Open"].iloc[0] == pytest.approx(999999.99)
    assert stored["Low"].iloc[0] == pytest.approx(0.01)
    assert stored["Volume"].iloc[0] == 2147483647
    assert isinstance(stored["Date"].iloc[0], datetime.date)


def test_load_prices_since_filters(store, sample_row):
    older = sample_row.copy()
    older["Date"] = [datetime.date(2024, 1, 2)]
    store.upsert(pd.concat([older, sample_row], ignore_index=True))

    assert len(store.load_prices()) == 2
    assert len(store.load_prices(since="2025-01-01")) == 1
