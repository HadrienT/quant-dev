import pytest
import textwrap
import re
from unittest.mock import MagicMock

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

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
    expected_query = """
        MERGE `test_project.test_dataset.main_table` AS main
        USING `test_project.test_dataset.temp_table` AS temp
        ON main.Date = temp.Date AND main.Ticker = temp.Ticker
        WHEN MATCHED THEN
          UPDATE SET
            Open = temp.Open,
            High = temp.High,
            Low = temp.Low,
            Close = temp.Close,
            Volume = temp.Volume
        WHEN NOT MATCHED THEN
          INSERT (Date, Ticker, Open, High, Low, Close, Volume)
          VALUES (temp.Date, temp.Ticker, temp.Open, temp.High, temp.Low, temp.Close, temp.Volume)
    """
    expected_query = textwrap.dedent(expected_query).strip()

    # Call the function
    merge_into_main_table(mock_client, temp_table_id, main_table_id)

    actual_query = mock_client.query.call_args[0][0]
    actual_query = textwrap.dedent(actual_query).strip()

    # Normalisation : remplace toutes les séquences d'espaces (y compris retours à la ligne) par un espace simple
    normalized_expected = re.sub(r"\s+", " ", expected_query).strip()
    normalized_actual = re.sub(r"\s+", " ", actual_query).strip()

    # Assertions
    assert (
        normalized_actual == normalized_expected
    ), f"Expected query:\n{normalized_expected}\n\nActual query:\n{normalized_actual}"
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
