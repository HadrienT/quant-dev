import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery


# Download the list of S&P 500 tickers
# def get_sp500_tickers():
#     """Fetches the list of S&P 500 tickers from Wikipedia."""
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     table = pd.read_html(url, header=0)
#     sp500_table = table[0]  # The main table is the first one
#     return sp500_table["Symbol"].tolist()

def get_sp500_tickers():
    return pd.read_csv("tickers.csv", header=None)[0].tolist()



# Download previous day's data
def download_previous_day_data(tickers):
    """
    Downloads daily data for the tickers from the previous day.
    """
    # start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    # end_date = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")

    current_date = datetime.now() - timedelta(days=1)
    while not pd.Timestamp(current_date).isoweekday() in range(
        1, 6
    ):  # Monday (1) to Friday (5)
        current_date -= timedelta(days=1)

    start_date = current_date.strftime("%Y-%m-%d")
    end_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        print(f"No data available for {start_date}.")
        return pd.DataFrame()  # Returns an empty DataFrame if no data is available
    # Restructure the data
    data = data.stack(level=0, future_stack=True).reset_index()
    columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    return data[columns]


# Download data for all tickers
def download_sp500_data(tickers, start_date="2000-01-01", end_date=None):
    """
    Downloads daily data for all S&P 500 tickers.
    """
    # Fetch the data via yfinance
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",  # Organizes the data by ticker
        threads=True,  # Parallel download
    )

    data = data.stack(level=0, future_stack=True).reset_index()  # Make tickers a column
    columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    return data[columns]


# Load data into BigQuery
def load_data_to_bigquery(df, table_id, project_id):
    """
    Loads a DataFrame into a BigQuery table.

    Arguments:
        df : pandas.DataFrame containing the data to load.
        table_id : Full ID of the BigQuery table (e.g., dataset.table_name).
        project_id : Google Cloud project ID.
    """
    client = bigquery.Client(project=project_id)

    # Convert the data to BigQuery format
    job = client.load_table_from_dataframe(df, table_id)

    # Wait for the job to complete
    job.result()
    print(f"Data has been loaded into {table_id}")


# Load data into a temporary table
def load_to_temp_table(client, df, temp_table_id):
    """
    Loads data into a temporary BigQuery table.
    """
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE"  # Overwrites existing data
    )
    job = client.load_table_from_dataframe(df, temp_table_id, job_config=job_config)
    job.result()
    print(f"Data loaded into temporary table {temp_table_id}")


# Merge data with the main table
def merge_into_main_table(client, temp_table_id, main_table_id):
    """
    Merges data from the temporary table into the main table.
    """
    query = f"""
    MERGE `{main_table_id}` AS main
    USING `{temp_table_id}` AS temp
    ON main.Date = temp.Date AND main.Ticker = temp.Ticker
    WHEN NOT MATCHED THEN
      INSERT (Date, Ticker, Open, High, Low, Close, Volume) 
      VALUES (temp.Date, temp.Ticker, temp.Open, temp.High, temp.Low, temp.Close, temp.Volume)
    """
    job = client.query(query)
    job.result()
    print(f"Data merged into main table {main_table_id}")


def fill_table():
    # Step 1: Fetch tickers
    print("Downloading S&P 500 tickers...")
    sp500_tickers = get_sp500_tickers()
    print(f"Number of tickers fetched: {len(sp500_tickers)}")

    # Step 2: Download data
    print("Downloading daily data...")
    end_date = datetime.today().strftime("%Y-%m-%d")
    sp500_data = download_sp500_data(
        sp500_tickers, start_date="2000-01-01", end_date=end_date
    )

    # Step 3: Load into BigQuery
    print("Loading data into BigQuery...")
    PROJECT_ID = "quant-dev-442615"  # Replace with your project ID
    DATASET_ID = "financial_data"  # Replace with your dataset name
    TABLE_ID = "sp500_data"  # Table name

    # Load data
    load_data_to_bigquery(
        sp500_data,
        table_id=f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",
        project_id=PROJECT_ID,
    )


def add_daily():
    # Step 1: Configuration
    PROJECT_ID = "quant-dev-442615"
    DATASET_ID = "financial_data"
    MAIN_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.sp500_data"
    TEMP_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.temp_sp500_data"

    # Fetch tickers
    print("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()

    # Step 2: Download data
    print("Downloading data for the previous day...")
    sp500_data = download_previous_day_data(tickers)
    if sp500_data.empty:
        print("No data available. Execution stopped.")
        return  # Stop execution if no data is available
    sp500_data["Date"] = pd.to_datetime(sp500_data["Date"]).dt.date
    # Convert the Volume column to integer
    sp500_data["Volume"] = sp500_data["Volume"].fillna(0).astype(int)

    # Step 3: Load into BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    print("Loading data into a temporary table...")
    load_to_temp_table(client, sp500_data, TEMP_TABLE_ID)

    # Step 4: Merge with the main table
    print("Merging data with the main table...")
    merge_into_main_table(client, TEMP_TABLE_ID, MAIN_TABLE_ID)


def main(request):
    add_daily()
    return "Success", 200


if __name__ == "__main__":
    fill_table()
    # add_daily()
