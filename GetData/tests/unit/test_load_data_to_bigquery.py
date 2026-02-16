import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import load_data_to_bigquery


@pytest.fixture
def sample_dataframe():
    """
    Returns a sample Pandas DataFrame with stock data for testing.
    """
    data = {
        "Date": [
            "2025-02-05",
            "2025-02-05",
            "2025-02-06",
            "2025-02-06",
            "2025-02-07",
            "2025-02-07",
        ],
        "Ticker": ["MSFT", "AAPL", "MSFT", "AAPL", "MSFT", "AAPL"],
        "Open": [412.35, 228.53, 414.00, 231.29, 416.48, 232.60],
        "High": [413.83, 232.67, 418.20, 233.80, 418.65, 234.00],
        "Low": [410.40, 228.27, 414.00, 230.43, 408.10, 227.26],
        "Close": [413.29, 232.47, 415.82, 233.22, 409.75, 227.63],
        "Volume": [16316700, 39620300, 16309800, 29925300, 22860700, 39666100],
    }
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@patch("google.cloud.bigquery.Client")  # Mock the BigQuery Client
def test_load_data_to_bigquery(mock_bigquery_client, sample_dataframe):
    """
    Unit test for load_data_to_bigquery function.
    Ensures that BigQuery API is called correctly.
    """
    # Mock BigQuery client instance
    mock_client_instance = MagicMock()
    mock_bigquery_client.return_value = mock_client_instance

    # Mock job result
    mock_job = MagicMock()
    mock_client_instance.load_table_from_dataframe.return_value = mock_job

    # Call the function
    project_id = "test-project"
    table_id = "test_dataset.test_table"
    load_data_to_bigquery(sample_dataframe, table_id, project_id)

    # Assertions
    mock_bigquery_client.assert_called_once_with(
        project=project_id
    )  # Ensure the client is created with the correct project ID
    mock_client_instance.load_table_from_dataframe.assert_called_once()
    call_args = mock_client_instance.load_table_from_dataframe.call_args
    assert call_args[0][0].equals(sample_dataframe)
    assert call_args[0][1] == table_id
    assert "job_config" in call_args[1]
    mock_job.result.assert_called_once()  # Ensure the job waits for completion
