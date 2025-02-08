import pytest
from unittest.mock import MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from main import merge_into_main_table


def test_merge_into_main_table():
    """
    Unit test for merge_into_main_table function.
    Tests if the correct MERGE query is executed and the job completion is verified.
    """
    # Mock BigQuery client
    mock_client = MagicMock()

    # Mock query job
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job

    # Test parameters
    temp_table_id = "test_project.test_dataset.temp_table"
    main_table_id = "test_project.test_dataset.main_table"

    # Expected query
    expected_query = f"""
    MERGE `{main_table_id}` AS main
    USING `{temp_table_id}` AS temp
    ON main.Date = temp.Date AND main.Ticker = temp.Ticker
    WHEN NOT MATCHED THEN
      INSERT (Date, Ticker, Open, High, Low, Close, Volume)
      VALUES (temp.Date, temp.Ticker, temp.Open, temp.High, temp.Low, temp.Close, temp.Volume)
    """

    # Call the function
    merge_into_main_table(mock_client, temp_table_id, main_table_id)

    # Assertions
    mock_client.query.assert_called_once_with(expected_query)
    mock_job.result.assert_called_once()  # Verify job completion was checked


def test_merge_into_main_table_error_handling():
    """
    Unit test for merge_into_main_table function error handling.
    Tests if errors during query execution are properly propagated.
    """
    # Mock BigQuery client
    mock_client = MagicMock()

    # Mock query job with error
    mock_job = MagicMock()
    mock_job.result.side_effect = Exception("Query failed")
    mock_client.query.return_value = mock_job

    # Test parameters
    temp_table_id = "test_project.test_dataset.temp_table"
    main_table_id = "test_project.test_dataset.main_table"

    # Verify that the exception is propagated
    with pytest.raises(Exception, match="Query failed"):
        merge_into_main_table(mock_client, temp_table_id, main_table_id)

    # Verify the query was attempted
    mock_client.query.assert_called_once()
