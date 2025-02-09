import pytest
import pandas as pd
from unittest.mock import patch


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import download_sp500_data


@pytest.fixture
def mock_sp500_data():
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


@patch("yfinance.download")
def test_download_sp500_data(mock_yf_download, mock_sp500_data):
    """
    Unit test for download_sp500_data function with a mocked yfinance API response.
    """
    mock_yf_download.return_value = mock_sp500_data  # Mock the API response

    tickers = ["AAPL", "MSFT"]
    start_date = "2024-01-01"
    end_date = "2024-02-07"

    result = download_sp500_data(tickers, start_date=start_date, end_date=end_date)

    # Ensure yfinance.download() was called with the correct parameters
    mock_yf_download.assert_called_once_with(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        threads=True,
    )

    # Validate the structure of the returned DataFrame
    assert not result.empty, "The returned DataFrame should not be empty"
    assert list(result.columns) == [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ], "Column names do not match expected format"

    # Ensure the tickers are correctly extracted
    assert set(result["Ticker"]) == {
        "AAPL",
        "MSFT",
    }, "Tickers are not correctly formatted"
