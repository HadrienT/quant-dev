import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from google.cloud import bigquery

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from main import load_to_temp_table


@pytest.fixture
def sample_dataframe():
    """
    Returns a sample Pandas DataFrame with stock data for testing.
    """
    data = {
        "Date": ["2025-02-05", "2025-02-06", "2025-02-07"],
        "Ticker": ["MSFT", "AAPL", "MSFT"],
        "Open": [412.35, 228.53, 414.00],
        "High": [413.83, 232.67, 418.20],
        "Low": [410.40, 228.27, 414.00],
        "Close": [413.29, 232.47, 415.82],
        "Volume": [16316700, 39620300, 16309800],
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@patch("google.cloud.bigquery.Client")
def test_load_to_temp_table(mock_bigquery_client, sample_dataframe):
    """
    Unit test for load_to_temp_table function.
    Ensures that BigQuery API is called correctly.
    """
    # Mock BigQuery client instance
    mock_client_instance = MagicMock()
    mock_bigquery_client.return_value = mock_client_instance

    # Mock job result
    mock_job = MagicMock()
    mock_client_instance.load_table_from_dataframe.return_value = mock_job

    # Define parameters
    temp_table_id = "test_project.test_dataset.temp_table"

    # Call the function
    load_to_temp_table(mock_client_instance, sample_dataframe, temp_table_id)

    # Assertions
    mock_client_instance.load_table_from_dataframe.assert_called_once()
    call_args = mock_client_instance.load_table_from_dataframe.call_args
    assert call_args[0][0].equals(sample_dataframe)  # Check DataFrame
    assert call_args[0][1] == temp_table_id  # Check table ID
    assert isinstance(
        call_args[1]["job_config"], (bigquery.LoadJobConfig, MagicMock)
    )  # Check job_config type

    mock_job.result.assert_called_once()  # Ensure the job waits for completion
