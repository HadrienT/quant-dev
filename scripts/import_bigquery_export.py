#!/usr/bin/env python3
"""Loads a BigQuery export of sp500_data into the local Postgres.

Step 2 of docs/decommission-runbook.md in the cloud-infra repository. The
BigQuery table is exported to Parquet (or CSV) on GCS, downloaded, and fed
through the same upsert path the daily ingestion uses, so the migrated rows go
through exactly the code that will maintain them afterwards.

    python scripts/import_bigquery_export.py ~/quant-dev-data/export
    python scripts/import_bigquery_export.py ~/quant-dev-data/export --expect-rows 3421887

Re-running is safe: the upsert is keyed on (date, ticker).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "GetData"))

from config import COLUMNS  # noqa: E402
from storage import PostgresStore  # noqa: E402

logger = logging.getLogger("quant-dev.import")

SUFFIXES = (".parquet", ".csv")


def read_export_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}; got {list(df.columns)}")

    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Ticker"] = df["Ticker"].astype(str)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="directory holding the exported parquet/csv files")
    parser.add_argument(
        "--expect-rows",
        type=int,
        help="row count reported by BigQuery; the import fails if the loaded table does not match",
    )
    parser.add_argument("--table", default=None, help="target table (defaults to MAIN_TABLE_NAME)")
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    files = sorted(p for p in args.export_dir.rglob("*") if p.suffix in SUFFIXES)
    if not files:
        logger.error("No %s files found under %s", " or ".join(SUFFIXES), args.export_dir)
        return 1
    logger.info("Found %s export files", len(files))

    store = PostgresStore(table=args.table) if args.table else PostgresStore()
    store.ensure_schema()

    total_sent = 0
    for path in files:
        df = read_export_file(path)
        total_sent += store.upsert(df)
        logger.info("%s: %s rows (%s sent so far)", path.name, len(df), total_sent)

    stored = store.row_count()
    first, last = store.date_range()
    logger.info("Table now holds %s rows, %s to %s", stored, first, last)

    # The stored count is normally below the sent count, because the export can
    # contain rows that collide on (date, ticker) and collapse into one.
    if args.expect_rows is not None and stored != args.expect_rows:
        logger.error(
            "Row count mismatch: BigQuery reported %s, Postgres holds %s. "
            "Do NOT destroy the BigQuery dataset.",
            args.expect_rows,
            stored,
        )
        return 1

    if args.expect_rows is not None:
        logger.info("Row count matches BigQuery (%s). The dataset is safe to destroy.", stored)
    else:
        logger.warning(
            "No --expect-rows given, so the migration was not verified against BigQuery. "
            "Re-run with --expect-rows before destroying the dataset."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
