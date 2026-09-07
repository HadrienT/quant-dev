import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from main import add_daily


@pytest.fixture
def sample_tickers():
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def sample_data():
    data = {
        "Date": ["2025-02-07", "2025-02-07", "2025-02-07"],
        "Ticker": ["MSFT", "AAPL", "GOOGL"],
        "Open": [412.35, 228.53, 414.00],
        "High": [413.83, 232.67, 418.20],
        "Low": [410.40, 228.27, 414.00],
        "Close": [413.29, 232.47, 415.82],
        "Volume": [16316700, 39620300, 16309800],
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Volume"] = df["Volume"].astype(int)
    return df


@pytest.fixture
def store():
    """A stand-in for PostgresStore, so no database is needed."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    return mock


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
def test_add_daily_successful_execution(mock_download, mock_get_tickers, store, sample_tickers, sample_data):
    """Tickers are fetched, the previous session is downloaded, then upserted."""
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = sample_data

    rows = add_daily(store)

    mock_get_tickers.assert_called_once()
    mock_download.assert_called_once_with(sample_tickers)

    store.ensure_schema.assert_called_once()
    store.upsert.assert_called_once()
    upserted = store.upsert.call_args[0][0]
    assert upserted.equals(sample_data)
    assert rows == 3


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
def test_add_daily_no_data_available(mock_download, mock_get_tickers, store, sample_tickers):
    """An empty download must not touch the database at all."""
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = pd.DataFrame()

    rows = add_daily(store)

    assert rows == 0
    store.ensure_schema.assert_not_called()
    store.upsert.assert_not_called()


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
def test_add_daily_error_handling(mock_download, mock_get_tickers, store, sample_tickers, sample_data):
    """A storage failure propagates rather than being swallowed."""
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = sample_data
    store.upsert.side_effect = Exception("Failed to load data")

    with pytest.raises(Exception, match="Failed to load data"):
        add_daily(store)

    store.ensure_schema.assert_called_once()
    store.upsert.assert_called_once()
