"""Environment-driven settings for the ingestion pipeline."""

import os
from pathlib import Path

# Postgres connection. Everything runs on the local server now, so there is no
# cloud project or credentials file involved.
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5433"))
PGDATABASE = os.getenv("PGDATABASE", "quantdev")
PGUSER = os.getenv("PGUSER", "quantdev")
PGPASSWORD = os.getenv("PGPASSWORD", "")

MAIN_TABLE_NAME = os.getenv("MAIN_TABLE_NAME", "sp500_data")

TICKERS_PATH = Path(os.getenv("TICKERS_PATH", Path(__file__).with_name("tickers.csv")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

MARKET_CALENDAR = os.getenv("MARKET_CALENDAR", "NYSE")
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# The column order every DataFrame in the pipeline uses. Capitalised because
# that is what yfinance produces; storage.py maps it to the lowercase Postgres
# column names.
COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


def dsn() -> str:
    """libpq connection string built from the PG* environment variables."""
    password = f" password={PGPASSWORD}" if PGPASSWORD else ""
    return f"host={PGHOST} port={PGPORT} dbname={PGDATABASE} user={PGUSER}{password}"
