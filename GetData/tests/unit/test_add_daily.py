import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

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


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
@patch("google.cloud.bigquery.Client")
@patch("main.load_to_temp_table")
@patch("main.merge_into_main_table")
def test_add_daily_successful_execution(
    mock_merge,
    mock_load,
    mock_bigquery_client,
    mock_download,
    mock_get_tickers,
    sample_tickers,
    sample_data,
):
    """
    Test successful execution of the add_daily function with all steps.
    """
    # Configure mocks
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = sample_data
    mock_client = MagicMock()
    mock_bigquery_client.return_value = mock_client

    # Execute function
    add_daily()

    # Verify all steps were executed in order
    mock_get_tickers.assert_called_once()
    mock_download.assert_called_once_with(sample_tickers)

    # Verify BigQuery client was created with correct project
    mock_bigquery_client.assert_called_once_with(project="quant-dev-442615")

    # Verify temp table load
    mock_load.assert_called_once()
    _, args, _ = mock_load.mock_calls[0]
    assert args[0] == mock_client  # client
    assert args[1].equals(sample_data)  # dataframe
    assert args[2] == "quant-dev-442615.financial_data.temp_sp500_data"  # temp table id

    # Verify merge operation
    mock_merge.assert_called_once_with(
        mock_client,
        "quant-dev-442615.financial_data.temp_sp500_data",
        "quant-dev-442615.financial_data.sp500_data",
    )


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
@patch("google.cloud.bigquery.Client")
@patch("main.load_to_temp_table")
@patch("main.merge_into_main_table")
def test_add_daily_no_data_available(
    mock_merge,
    mock_load,
    mock_bigquery_client,
    mock_download,
    mock_get_tickers,
    sample_tickers,
):
    """
    Test early exit when no data is available.
    """
    # Configure mocks
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = pd.DataFrame()  # Empty DataFrame

    # Execute function
    add_daily()

    # Verify early exit
    mock_get_tickers.assert_called_once()
    mock_download.assert_called_once()
    mock_bigquery_client.assert_not_called()
    mock_load.assert_not_called()
    mock_merge.assert_not_called()


@patch("main.get_sp500_tickers")
@patch("main.download_previous_day_data")
@patch("google.cloud.bigquery.Client")
@patch("main.load_to_temp_table")
@patch("main.merge_into_main_table")
def test_add_daily_error_handling(
    mock_merge,
    mock_load,
    mock_bigquery_client,
    mock_download,
    mock_get_tickers,
    sample_tickers,
    sample_data,
):
    """
    Test error handling during execution.
    """
    # Configure mocks
    mock_get_tickers.return_value = sample_tickers
    mock_download.return_value = sample_data
    mock_client = MagicMock()
    mock_bigquery_client.return_value = mock_client
    mock_load.side_effect = Exception("Failed to load data")

    # Verify exception is propagated
    with pytest.raises(Exception, match="Failed to load data"):
        add_daily()

    # Verify steps until error
    mock_get_tickers.assert_called_once()
    mock_download.assert_called_once()
    mock_load.assert_called_once()
    mock_merge.assert_not_called()  # Should not be called after error
