# RiskFlux

Production-grade ML pipeline for loan default prediction on Lending Club data (2007–2020Q3).

The model is intentionally simple (LightGBM). The focus is the infrastructure around it:
reproducible data + pipeline versioning, train/serve parity, experiment tracking, a model
registry with promotion, a containerized inference API, drift monitoring, automated
retraining, and a model card.

## Status
🚧 Step 4 done — shared feature pipeline + train/serve parity test

## Stack
Python 3.11 · uv · DVC · pandera · LightGBM · MLflow · FastAPI · Docker · Evidently · GitHub Actions · GCP Cloud Run

## Setup
```bash
uv sync                 # create .venv from uv.lock
uv run dvc pull         # fetch versioned raw data from DagsHub (needs a DagsHub token, see below)
```
DagsHub credentials (stored in `.dvc/config.local`, never committed):
```bash
uv run dvc remote modify origin --local access_key_id <DAGSHUB_TOKEN>
uv run dvc remote modify origin --local secret_access_key <DAGSHUB_TOKEN>
```

## Pipeline (run from repo root)
```bash
uv run python -m riskflux.data.ingest   # raw zip -> data/interim/loans_raw.parquet (all strings)
uv run python -m riskflux.data.clean    # -> loans_clean.parquet + loans_aux.parquet (validated)
uv run python -m riskflux.data.split    # -> data/processed/{train,val,test}.parquet
uv run pytest                           # unit tests (synthetic data, no download needed)
```
Parameters (population window, split cutoffs) live in `params.yaml`. Step 5 wires these stages into `dvc repro`.

## Data
- Source: [Lending Club 2007–2020Q3 on Kaggle](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
- `scripts/download_data.sh` is a one-time bootstrap; the DVC remote is the source of truth.
- Population: 36-month individual loans issued 2013-01 … 2017-04 with a final outcome — 953,889 loans, 14.7% default.

## Design decisions
See [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/COLUMN_AUDIT.md](docs/COLUMN_AUDIT.md).
