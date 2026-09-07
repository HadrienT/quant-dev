# Quantitative Finance Project

> **This repository is largely an archive.** The pieces that are still
> developed live elsewhere:
>
> - **Data ingestion** → [`data-ingest`](https://github.com/HadrienT/data-ingest),
>   which owns the pipelines and the Postgres they write to.
> - **Cloud infrastructure** → [`cloud-infra`](https://github.com/HadrienT/cloud-infra) (private).
> - **Pricing and modelling** → [`quant-modeling`](https://github.com/HadrienT/quant-modeling).
>
> What remains here is the Streamlit visualisation app, kept running locally
> against the data-ingest database.

A personal project exploring quantitative finance: portfolio construction and
analysis over daily market data.

## What is left here

- **visualization** — a [Streamlit](https://streamlit.io/) app for portfolio
  construction (stocks, bonds, ETFs, crypto), static and Sharpe-optimised
  dynamic allocation, implied volatility curves and surfaces, and a pair
  trading module still under development. Local only; it is no longer
  deployed publicly.

## Running it

The database belongs to the data-ingest stack, so start that first:

```bash
cd ~/data-ingest && docker compose up -d postgres
```

Then, here:

```bash
cp .env.example .env    # values must match data-ingest's .env
$EDITOR .env
docker compose up -d
```

The app is on <http://127.0.0.1:8501>. It joins data-ingest's compose network
and reaches Postgres by service name, so the database publishes no port beyond
the host loopback.

It reads `prices.sp500_daily`; override `PRICES_SCHEMA` and `PRICES_TABLE` to
point it somewhere else.

## History

This started as a single repository holding the GCP infrastructure, an
ingestion pipeline writing to BigQuery, and this app deployed on Cloud Run.
Everything moved to a self-hosted server: the cloud resources were
decommissioned, the ingestion became its own project, and only the app stayed.

## Known gaps

- The architecture diagram on the app's Architecture page
  (`visualization/assets/quant-dev 3.svg`) still shows the old GCP topology and
  needs redrawing.

## License

MIT. See the LICENSE file.

## Contact

tramonihadrien@gmail.com
