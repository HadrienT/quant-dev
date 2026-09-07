# Quantitative Finance Project

A personal project exploring quantitative finance: daily market data ingestion,
portfolio construction and analysis, and interactive visualisation.

Everything runs on a self-hosted server. It previously ran on Google Cloud
(Cloud Scheduler → Cloud Function → BigQuery, with the app on Cloud Run); that
infrastructure is being decommissioned and what is left of it now lives in the
separate [`cloud-infra`](https://github.com/HadrienT/cloud-infra) repository.

## Repository structure

- **GetData** — the ingestion pipeline. Pulls daily OHLCV data for the S&P 500
  from [yfinance](https://pypi.org/project/yfinance/) and upserts it into
  Postgres. Runs as a container fired by a systemd timer.
- **visualization** — a [Streamlit](https://streamlit.io/) app for portfolio
  construction (stocks, bonds, ETFs, crypto), static and Sharpe-optimised
  dynamic allocation, implied volatility curves and surfaces, and a pair
  trading module still under development. Local only; it is no longer deployed
  publicly.
- **scripts** — the systemd units for the ingestion timer, and the one-off
  importer used to migrate the BigQuery table into Postgres.

## Architecture

```
                 systemd timer (Mon-Fri 22:00)
                            │
                            ▼
  yfinance ──────▶ GetData ingestion ──────▶ Postgres ──────▶ Streamlit app
                   (container, one-shot)     (container,      (container,
                                              volume pgdata)   127.0.0.1:8501)
```

Every port binds to `127.0.0.1`, so nothing is reachable from outside the
server.

## Getting started

Requirements: Docker with the Compose plugin, and Python 3.11 to run the tests
outside a container.

```bash
cp .env.example .env
$EDITOR .env            # set PGPASSWORD, and FRED_API_KEY if you want the app

docker compose up -d postgres
```

Backfill the full history, then check what landed:

```bash
docker compose run --rm ingestion --mode full
docker compose run --rm ingestion --status
```

Schedule the daily run — this replaces the Cloud Scheduler job:

```bash
./scripts/install-timer.sh
```

Start the Streamlit app on <http://127.0.0.1:8501>:

```bash
docker compose --profile viz up -d
```

### Migrating from the old BigQuery table

`scripts/import_bigquery_export.py` loads a BigQuery export through the same
upsert path the daily ingestion uses. Pass the row count BigQuery reports so the
migration is actually verified rather than assumed:

```bash
python scripts/import_bigquery_export.py ~/quant-dev-data/export --expect-rows 3421887
```

The full sequence, and the order in which the GCP services are torn down, is in
[`cloud-infra/docs/decommission-runbook.md`](https://github.com/HadrienT/cloud-infra/blob/main/docs/decommission-runbook.md).

## Tests

```bash
cd GetData
pip install -r requirements.txt -r requirements-dev.in
pytest -m "not network"          # unit tests, no database and no network needed
pytest                           # adds the live yfinance tests
```

Tests marked `postgres` skip themselves when no database is reachable; tests
marked `network` call the live yfinance API and are excluded in CI, where the
API's rate limits and shifting response shape make them unreliable.

## Known gaps

- The architecture diagram on the app's Architecture page
  (`visualization/assets/quant-dev 3.svg`) still shows the old GCP topology and
  needs redrawing.
- The app's Postgres credentials come from the environment; there is no
  read-only role yet, so it connects as the same user the ingestion writes with.

## License

MIT. See the LICENSE file.

## Contact

tramonihadrien@gmail.com
