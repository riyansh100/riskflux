# RiskFlux

Production-grade ML pipeline for loan default prediction on Lending Club data (2007–2020Q3).

The model is intentionally simple (LightGBM). The focus is the infrastructure around it:
reproducible data + pipeline versioning, train/serve parity, experiment tracking, a model
registry with promotion, a containerized inference API, drift monitoring, automated
retraining, and a model card.

## Status
🚧 Step 0 — project setup

## Stack
Python 3.11 · uv · DVC · LightGBM · MLflow · FastAPI · Docker · Evidently · GitHub Actions · GCP Cloud Run

## Design decisions
See [docs/DECISIONS.md](docs/DECISIONS.md).