"""Daily ingestion of S&P 500 OHLCV data into the local Postgres.

Previously a GCP Cloud Function triggered by Cloud Scheduler and writing to
BigQuery. It is now a plain CLI, run on the local server by a systemd timer:

    python main.py --mode daily   # previous trading day, upserted
    python main.py --mode full    # full history since 2000, one-off backfill
    python main.py --status       # row count and date range
"""

import argparse
import logging
import sys

from config import LOG_LEVEL
from market_data import (  # noqa: F401 - re-exported for callers and tests
    _last_trading_day,
    download_previous_day_data,
    download_sp500_data,
    get_sp500_tickers,
)
from storage import PostgresStore

logger = logging.getLogger("quant-dev.ingestion")


def fill_table(store: PostgresStore = None) -> int:
    """One-off backfill of the full history."""
    store = store or PostgresStore()

    logger.info("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()
    logger.info("Number of tickers fetched: %s", len(tickers))

    logger.info("Downloading full daily history...")
    sp500_data = download_sp500_data(tickers)
    if sp500_data.empty:
        logger.warning("No data available for full ingestion.")
        return 0

    store.ensure_schema()
    logger.info("Loading full history into Postgres...")
    return store.upsert(sp500_data)


def add_daily(store: PostgresStore = None) -> int:
    """Upserts the previous trading day. This is what the timer runs."""
    store = store or PostgresStore()

    logger.info("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()

    logger.info("Downloading data for the previous trading day...")
    sp500_data = download_previous_day_data(tickers)
    if sp500_data.empty:
        logger.warning("No data available. Execution stopped.")
        return 0

    store.ensure_schema()
    logger.info("Upserting into the main table...")
    return store.upsert(sp500_data)


def status(store: PostgresStore = None) -> None:
    store = store or PostgresStore()
    # Reporting on a database that has never been ingested into is a normal
    # first-run case, not an error.
    store.ensure_schema()
    first, last = store.date_range()
    logger.info("Rows: %s | first session: %s | last session: %s", store.row_count(), first, last)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S&P 500 daily ingestion")
    parser.add_argument("--mode", choices=["daily", "full"], default="daily")
    parser.add_argument("--status", action="store_true", help="report table stats and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

    if args.status:
        status()
        return 0

    rows = fill_table() if args.mode == "full" else add_daily()
    logger.info("Done, %s rows written", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
