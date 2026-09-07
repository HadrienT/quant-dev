"""Postgres storage for the daily OHLCV data.

Replaces the BigQuery layer the pipeline used while it ran on GCP. The write
path keeps the same two-step shape as the old BigQuery MERGE: rows land in a
staging table first, then a single upsert moves them into the main table, so a
partial download can never leave the main table half-updated.
"""

import logging
from contextlib import contextmanager
from datetime import date
from typing import Iterator, Optional, Tuple

import pandas as pd
import psycopg
from psycopg import sql

from config import COLUMNS, MAIN_TABLE_NAME, dsn

logger = logging.getLogger("quant-dev.ingestion")

# DataFrames carry yfinance's capitalised names; Postgres columns are lowercase.
DB_COLUMNS = [col.lower() for col in COLUMNS]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    date    DATE             NOT NULL,
    ticker  TEXT             NOT NULL,
    open    DOUBLE PRECISION,
    high    DOUBLE PRECISION,
    low     DOUBLE PRECISION,
    close   DOUBLE PRECISION,
    volume  BIGINT,
    PRIMARY KEY (date, ticker)
)
"""

# The main read pattern is "one ticker over a date range", which the primary key
# on (date, ticker) does not serve well.
CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS {index} ON {table} (ticker, date)
"""


class PostgresStore:
    """Reads and writes the OHLCV table."""

    def __init__(self, connection_string: Optional[str] = None, table: str = MAIN_TABLE_NAME):
        self.dsn = connection_string or dsn()
        self.table = table

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.dsn) as conn:
            yield conn

    def _ident(self) -> sql.Identifier:
        return sql.Identifier(self.table)

    def ensure_schema(self) -> None:
        """Creates the table and its index if they do not exist yet."""
        with self.connect() as conn:
            conn.execute(sql.SQL(CREATE_TABLE).format(table=self._ident()))
            conn.execute(
                sql.SQL(CREATE_INDEX).format(
                    index=sql.Identifier(f"{self.table}_ticker_date_idx"),
                    table=self._ident(),
                )
            )
        logger.info("Schema ready for table %s", self.table)

    def upsert(self, df: pd.DataFrame) -> int:
        """Inserts rows, updating any that already exist for (date, ticker).

        Returns the number of rows sent. Mirrors the semantics of the BigQuery
        MERGE this replaces: last write wins on a conflict.
        """
        if df.empty:
            logger.info("Nothing to upsert")
            return 0

        frame = df[[col for col in COLUMNS if col in df.columns]].copy()
        missing = set(COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

        columns = sql.SQL(", ").join(sql.Identifier(col) for col in DB_COLUMNS)
        updatable = [col for col in DB_COLUMNS if col not in ("date", "ticker")]
        assignments = sql.SQL(", ").join(
            sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col)) for col in updatable
        )

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "CREATE TEMP TABLE staging (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP"
                    ).format(table=self._ident())
                )

                copy_stmt = sql.SQL("COPY staging ({columns}) FROM STDIN").format(columns=columns)
                with cur.copy(copy_stmt) as copy:
                    for row in frame.itertuples(index=False, name=None):
                        copy.write_row(row)

                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table} ({columns})
                        SELECT {columns} FROM staging
                        ON CONFLICT (date, ticker) DO UPDATE SET {assignments}
                        """
                    ).format(table=self._ident(), columns=columns, assignments=assignments)
                )
                sent = len(frame)

        logger.info("Upserted %s rows into %s", sent, self.table)
        return sent

    def row_count(self) -> int:
        with self.connect() as conn:
            result = conn.execute(sql.SQL("SELECT COUNT(*) FROM {table}").format(table=self._ident()))
            return result.fetchone()[0]

    def date_range(self) -> Tuple[Optional[date], Optional[date]]:
        """Oldest and newest session in the table, or (None, None) if empty."""
        with self.connect() as conn:
            result = conn.execute(
                sql.SQL("SELECT MIN(date), MAX(date) FROM {table}").format(table=self._ident())
            )
            return result.fetchone()

    def load_prices(self, since: Optional[str] = None) -> pd.DataFrame:
        """Reads the table back as a DataFrame with the pipeline's column names."""
        query = sql.SQL("SELECT {columns} FROM {table}").format(
            columns=sql.SQL(", ").join(sql.Identifier(col) for col in DB_COLUMNS),
            table=self._ident(),
        )
        params: tuple = ()
        if since:
            query = sql.SQL("{query} WHERE date > %s").format(query=query)
            params = (since,)
        query = sql.SQL("{query} ORDER BY date").format(query=query)

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return pd.DataFrame(rows, columns=COLUMNS)
