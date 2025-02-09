import pytest
import pandas as pd
from google.cloud import bigquery
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../.."))

from main import (
    get_sp500_tickers,
    download_previous_day_data,
    load_to_temp_table,
    merge_into_main_table,
)


class TestE2E:
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
