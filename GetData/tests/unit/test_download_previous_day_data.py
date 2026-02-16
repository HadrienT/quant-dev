import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import download_previous_day_data


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame with a MultiIndex matching the expected structure."""
    data = {
        ("AAPL", "Open"): [130.788116],
        ("AAPL", "High"): [132.382025],
        ("AAPL", "Low"): [128.030558],
        ("AAPL", "Close"): [131.169479],
        ("AAPL", "Volume"): [106239800],
        ("MSFT", "Open"): [227.213221],
        ("MSFT", "High"): [234.404861],
        ("MSFT", "Low"): [224.671011],
        ("MSFT", "Close"): [231.649994],
        ("MSFT", "Volume"): [33314200],
    }

    index = pd.to_datetime(["2021-02-01"])  # Sample date
    df = pd.DataFrame(data, index=index)

    # Convert to MultiIndex
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Price"])
    df.index.name = "Date"  # Name the index correctly
    return df


@patch("main._last_trading_day")
@patch("yfinance.download")
def test_download_previous_day_data(mock_yf_download, mock_last_trading_day, sample_data):
    """Test function download_previous_day_data with simulated data."""
    fixed_date = datetime(2024, 2, 6)
    mock_last_trading_day.return_value = fixed_date
    mock_yf_download.return_value = sample_data  # Mock yfinance.download()

    tickers = ["AAPL", "MSFT"]
    result = download_previous_day_data(tickers)

    # Check the arguments passed to yfinance.download()
    expected_start_date = fixed_date.strftime("%Y-%m-%d")
    expected_end_date = (fixed_date + timedelta(days=1)).strftime("%Y-%m-%d")

    mock_yf_download.assert_called_once_with(
        tickers=" ".join(tickers),
        start=expected_start_date,
        end=expected_end_date,
        interval="1d",
        group_by="ticker",
        threads=True,
    )

    # Check the structure of the returned DataFrame
    assert not result.empty, "Le DataFrame retourné ne doit pas être vide"
    assert list(result.columns) == [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ], "Columns do not match the expected structure."

    # Check the levels of the MultiIndex
    assert set(result["Ticker"]) == {
        "AAPL",
        "MSFT",
    }, "Incorrect tickers in the DataFrame"


@patch("main._last_trading_day")
@patch("yfinance.download")
def test_download_previous_day_data_no_data(mock_yf_download, mock_last_trading_day):
    """Test the case where no data is returned."""
    mock_last_trading_day.return_value = datetime(2024, 2, 6)
    mock_yf_download.return_value = pd.DataFrame()  # Simulate no data returned

    tickers = ["AAPL", "MSFT"]
    result = download_previous_day_data(tickers)

    # Check that the API was called
    mock_yf_download.assert_called()

    # Check that the result is an empty DataFrame
    assert result.empty, "DataFrame should be empty if no data is available"
