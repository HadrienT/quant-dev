import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from google.cloud import bigquery

logger = logging.getLogger("quant-dev.ingestion")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

PROJECT_ID = os.getenv("PROJECT_ID", "quant-dev-442615")
DATASET_ID = os.getenv("DATASET_ID", "financial_data")
MAIN_TABLE_NAME = os.getenv("MAIN_TABLE_NAME", "sp500_data")
TEMP_TABLE_NAME = os.getenv("TEMP_TABLE_NAME", "temp_sp500_data")
DATASET_LOCATION = os.getenv("DATASET_LOCATION", "US")
TICKERS_PATH = Path(os.getenv("TICKERS_PATH", Path(__file__).with_name("tickers.csv")))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))
MARKET_CALENDAR = os.getenv("MARKET_CALENDAR", "NYSE")
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

SCHEMA = [
    bigquery.SchemaField("Date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("Ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("Open", "FLOAT64"),
    bigquery.SchemaField("High", "FLOAT64"),
    bigquery.SchemaField("Low", "FLOAT64"),
    bigquery.SchemaField("Close", "FLOAT64"),
    bigquery.SchemaField("Volume", "INTEGER"),
]


def get_sp500_tickers() -> List[str]:
    return pd.read_csv(TICKERS_PATH, header=None)[0].dropna().astype(str).tolist()


def _chunked(items: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _last_trading_day(reference_dt: Optional[datetime] = None, calendar_name: str = MARKET_CALENDAR) -> datetime:
    now = reference_dt or datetime.now(ZoneInfo(MARKET_TZ))
    start = (now - timedelta(days=10)).date()
    end = now.date()
    try:
        calendar = mcal.get_calendar(calendar_name)
        schedule = calendar.schedule(start_date=start, end_date=end)
        if schedule.empty:
            raise ValueError("Empty market schedule")
        last_session = schedule.index[-1].to_pydatetime().date()
        return datetime.combine(last_session, datetime.min.time(), tzinfo=ZoneInfo(MARKET_TZ))
    except Exception as exc:
        logger.warning("Falling back to weekday logic: %s", exc)
        current_date = now - timedelta(days=1)
        while current_date.isoweekday() not in range(1, 6):
            current_date -= timedelta(days=1)
        return current_date


def _download_yfinance_chunk(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                start=start_date,
                end=end_date,
                interval="1d",
                group_by="ticker",
                threads=True,
            )
            if not data.empty:
                return data
        except Exception as exc:
            last_error = exc
            logger.warning("Download attempt %s failed: %s", attempt, exc)
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    if last_error:
        logger.error("Download failed after %s attempts: %s", MAX_RETRIES, last_error)
    return pd.DataFrame()


def _normalize_yfinance_data(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        data = df.stack(level=0, future_stack=True).reset_index()
    else:
        data = df.reset_index().copy()
        ticker = tickers[0] if tickers else "UNKNOWN"
        data["Ticker"] = ticker

    expected_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    data = data[[col for col in expected_cols if col in data.columns]]
    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].fillna(0).astype("int64")
    return data


def _download_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for batch in _chunked(tickers, CHUNK_SIZE):
        raw = _download_yfinance_chunk(batch, start_date, end_date)
        normalized = _normalize_yfinance_data(raw, batch)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["Date", "Ticker"])
    data = data.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    return data


def download_previous_day_data(tickers: List[str], reference_dt: Optional[datetime] = None) -> pd.DataFrame:
    """
    Downloads daily data for the tickers from the previous trading day.
    """
    last_session = _last_trading_day(reference_dt)
    start_date = last_session.strftime("%Y-%m-%d")
    end_date = (last_session + timedelta(days=1)).strftime("%Y-%m-%d")
    data = _download_prices(tickers, start_date, end_date)
    if data.empty:
        logger.info("No data available for %s", start_date)
    return data


def download_sp500_data(tickers: List[str], start_date: str = "2000-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Downloads daily data for all S&P 500 tickers.
    """
    end_date = end_date or datetime.now(ZoneInfo(MARKET_TZ)).strftime("%Y-%m-%d")
    return _download_prices(tickers, start_date, end_date)


def ensure_dataset_and_table(client: bigquery.Client, dataset_id: str, table_id: str) -> None:
    dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset_ref.location = DATASET_LOCATION
    client.create_dataset(dataset_ref, exists_ok=True)

    table_ref = bigquery.Table(table_id, schema=SCHEMA)
    table_ref.time_partitioning = bigquery.TimePartitioning(field="Date")
    table_ref.clustering_fields = ["Ticker"]
    client.create_table(table_ref, exists_ok=True)


def load_data_to_bigquery(df: pd.DataFrame, table_id: str, project_id: str) -> None:
    """
    Loads a DataFrame into a BigQuery table.
    """
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(schema=SCHEMA)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logger.info("Data has been loaded into %s", table_id)


def load_to_temp_table(client: bigquery.Client, df: pd.DataFrame, temp_table_id: str) -> None:
    """
    Loads data into a temporary BigQuery table.
    """
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=SCHEMA,
    )
    job = client.load_table_from_dataframe(df, temp_table_id, job_config=job_config)
    job.result()
    logger.info("Data loaded into temporary table %s", temp_table_id)


def merge_into_main_table(client: bigquery.Client, temp_table_id: str, main_table_id: str) -> None:
    """
    Upserts data from the temporary table into the main table.
    """
    query = f"""
    MERGE `{main_table_id}` AS main
    USING `{temp_table_id}` AS temp
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
    job = client.query(query)
    job.result()
    logger.info("Data merged into main table %s", main_table_id)


def fill_table() -> None:
    logger.info("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()
    logger.info("Number of tickers fetched: %s", len(tickers))

    logger.info("Downloading full daily history...")
    sp500_data = download_sp500_data(tickers)
    if sp500_data.empty:
        logger.warning("No data available for full ingestion.")
        return

    client = bigquery.Client(project=PROJECT_ID)
    main_table_id = f"{PROJECT_ID}.{DATASET_ID}.{MAIN_TABLE_NAME}"
    ensure_dataset_and_table(client, DATASET_ID, main_table_id)

    logger.info("Loading full history into BigQuery...")
    load_data_to_bigquery(sp500_data, main_table_id, PROJECT_ID)


def add_daily() -> None:
    main_table_id = f"{PROJECT_ID}.{DATASET_ID}.{MAIN_TABLE_NAME}"
    temp_table_id = f"{PROJECT_ID}.{DATASET_ID}.{TEMP_TABLE_NAME}"

    logger.info("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()

    logger.info("Downloading data for the previous trading day...")
    sp500_data = download_previous_day_data(tickers)
    if sp500_data.empty:
        logger.warning("No data available. Execution stopped.")
        return

    client = bigquery.Client(project=PROJECT_ID)
    ensure_dataset_and_table(client, DATASET_ID, main_table_id)

    logger.info("Loading data into a temporary table...")
    load_to_temp_table(client, sp500_data, temp_table_id)

    logger.info("Merging data with the main table...")
    merge_into_main_table(client, temp_table_id, main_table_id)


def main(request):
    mode = "daily"
    if request is not None:
        mode = request.args.get("mode", mode) if hasattr(request, "args") else mode
    if mode == "full":
        fill_table()
    else:
        add_daily()
    return "Success", 200


if __name__ == "__main__":
    add_daily()
