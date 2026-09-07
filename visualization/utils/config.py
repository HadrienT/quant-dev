import os

import QuantLib as ql
from dotenv import load_dotenv

load_dotenv()

fred_api_key = os.getenv("FRED_API_KEY")

# Postgres on the local server, replacing BigQuery. No credentials file and no
# cloud project are involved any more.
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5433"))
PGDATABASE = os.getenv("PGDATABASE", "dataingest")
PGUSER = os.getenv("PGUSER", "dataingest")
PGPASSWORD = os.getenv("PGPASSWORD", "")

# The data-ingest project owns this table: one schema per domain.
PRICES_SCHEMA = os.getenv("PRICES_SCHEMA", "prices")
PRICES_TABLE = os.getenv("PRICES_TABLE", "sp500_daily")

# The app only charts recent history; loading everything back to 2000 would
# push a few hundred MB into the Streamlit cache for no benefit.
HISTORY_START = os.getenv("HISTORY_START", "2020-01-01")


def dsn() -> str:
    """libpq connection string built from the PG* environment variables."""
    password = f" password={PGPASSWORD}" if PGPASSWORD else ""
    return f"host={PGHOST} port={PGPORT} dbname={PGDATABASE} user={PGUSER}{password}"


calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
