import pytest
import pandas as pd
from google.cloud import bigquery
import datetime
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import (
    get_sp500_tickers,
    download_previous_day_data,
    load_to_temp_table,
    merge_into_main_table,
)


class TestIntegration:
    @pytest.fixture(scope="class")
    def bigquery_client(self):
        """Creates a BigQuery client."""
        return bigquery.Client()

    @pytest.fixture(scope="class")
    def test_tables(self, bigquery_client):
        """
        Creates test tables and cleans them up after tests.
        """
        project_id = "quant-dev-442615"
        dataset_id = "financial_data_test"
        temp_table_id = f"{project_id}.{dataset_id}.temp_sp500_data_test"
        main_table_id = f"{project_id}.{dataset_id}.sp500_data_test"

        # Create test dataset if it doesn't exist
        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset.location = "EU"
        try:
            bigquery_client.create_dataset(dataset, exists_ok=True)
        except Exception as e:
            pytest.skip(f"Failed to create test dataset: {e}")

        # Create main table with schema
        schema = [
            bigquery.SchemaField("Date", "DATE"),
            bigquery.SchemaField("Ticker", "STRING"),
            bigquery.SchemaField("Open", "FLOAT64"),
            bigquery.SchemaField("High", "FLOAT64"),
            bigquery.SchemaField("Low", "FLOAT64"),
            bigquery.SchemaField("Close", "FLOAT64"),
            bigquery.SchemaField("Volume", "INTEGER"),
        ]

        table = bigquery.Table(main_table_id, schema=schema)
        try:
            bigquery_client.create_table(table, exists_ok=True)
        except Exception as e:
            pytest.skip(f"Failed to create test table: {e}")

        yield {"temp_table_id": temp_table_id, "main_table_id": main_table_id}

        # Cleanup after tests
        try:
            bigquery_client.delete_table(temp_table_id, not_found_ok=True)
            bigquery_client.delete_table(main_table_id, not_found_ok=True)
            bigquery_client.delete_dataset(
                dataset_id, delete_contents=True, not_found_ok=True
            )
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def test_end_to_end_pipeline(self, bigquery_client, test_tables):
        """
        Tests the entire pipeline from data download to BigQuery insertion.
        """
        # Step 1: Get tickers (limit to 5 for testing)
        tickers = get_sp500_tickers()[:5]
        assert len(tickers) > 0, "Failed to fetch tickers"

        # Step 2: Download data
        data = download_previous_day_data(tickers)
        assert not data.empty, "Failed to download stock data"
        assert all(
            col in data.columns
            for col in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        )
        data["Date"] = pd.to_datetime(data["Date"]).dt.date

        # Step 3: Load to temp table
        load_to_temp_table(bigquery_client, data, test_tables["temp_table_id"])

        # Verify temp table data
        query = f"SELECT COUNT(*) as count FROM `{test_tables['temp_table_id']}`"
        temp_count = (
            bigquery_client.query(query).result().to_dataframe()["count"].iloc[0]
        )
        assert temp_count == len(data), "Temp table row count doesn't match input data"

        # Step 4: Merge into main table
        merge_into_main_table(
            bigquery_client, test_tables["temp_table_id"], test_tables["main_table_id"]
        )

        # Verify main table data
        query = f"""
        SELECT COUNT(*) as count
        FROM `{test_tables['main_table_id']}`
        WHERE Date = '{data['Date'].iloc[0]}'
        """
        main_count = (
            bigquery_client.query(query).result().to_dataframe()["count"].iloc[0]
        )
        assert main_count > 0, "No data found in main table after merge"

    def test_duplicate_data_handling(self, bigquery_client, test_tables):
        """
        Tests that duplicate data is handled correctly during merge.
        """
        # Create sample data
        sample_data = pd.DataFrame(
            {
                "Date": [datetime.date.today()] * 2,
                "Ticker": ["AAPL", "AAPL"],  # Duplicate ticker
                "Open": [150.0, 150.0],
                "High": [155.0, 155.0],
                "Low": [149.0, 149.0],
                "Close": [153.0, 153.0],
                "Volume": [1000000, 1000000],
            }
        )
        sample_data["Date"] = pd.to_datetime(sample_data["Date"]).dt.date

        # Load data twice
        load_to_temp_table(bigquery_client, sample_data, test_tables["temp_table_id"])
        merge_into_main_table(
            bigquery_client, test_tables["temp_table_id"], test_tables["main_table_id"]
        )

        # Try to load the same data again
        load_to_temp_table(bigquery_client, sample_data, test_tables["temp_table_id"])
        merge_into_main_table(
            bigquery_client, test_tables["temp_table_id"], test_tables["main_table_id"]
        )

        # Verify no duplicates in main table
        query = f"""
        SELECT COUNT(*) as count
        FROM `{test_tables['main_table_id']}`
        WHERE Date = '{sample_data['Date'].iloc[0]}'
        AND Ticker = 'AAPL'
        """
        result_data = bigquery_client.query(query).result().to_dataframe()
        print(f"Result data: {result_data}")
        count = result_data["count"].iloc[0]
        assert count == 1, "Duplicate records found in main table"

    def test_data_types_and_constraints(self, bigquery_client, test_tables):
        """
        Tests that data types and constraints are properly enforced.
        """
        # Test data with various edge cases
        sample_data = pd.DataFrame(
            {
                "Date": [datetime.date.today()],
                "Ticker": ["AAPL"],
                "Open": [999999.99],  # Large number
                "High": [999999.99],
                "Low": [0.01],  # Small number
                "Close": [500.00],
                "Volume": [2147483647],  # Max INT32
            }
        )

        # Load and merge data
        load_to_temp_table(bigquery_client, sample_data, test_tables["temp_table_id"])
        merge_into_main_table(
            bigquery_client, test_tables["temp_table_id"], test_tables["main_table_id"]
        )

        # Verify data types and values
        query = f"""
        SELECT *
        FROM `{test_tables['main_table_id']}`
        WHERE Date = '{sample_data['Date'].iloc[0]}'
        AND Ticker = 'AAPL'
        """
        result = bigquery_client.query(query).result().to_dataframe()

        assert not result.empty, "No data found in main table"
        assert result["Open"].dtype == "float64", "Incorrect data type for Open"
        assert pd.api.types.is_integer_dtype(
            result["Volume"]
        ), "Incorrect data type for Volume"
        assert result["Open"].iloc[0] == 999999.99, "Large number not handled correctly"
        assert result["Low"].iloc[0] == 0.01, "Small number not handled correctly"
