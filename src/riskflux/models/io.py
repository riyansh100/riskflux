"""Reading splits and writing/reading the model + decision policy."""

import json
from pathlib import Path

import joblib
import pandas as pd

from riskflux import columns as C
from riskflux.config import AUX_PARQUET, MODEL_PATH, POLICY_PATH, SPLIT_DIR


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLIT_DIR / f"{name}.parquet")


def aux_for(loans: pd.DataFrame, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aux rows (LC decisions, outcomes) aligned row-for-row with `loans`."""
    aux = pd.read_parquet(AUX_PARQUET) if aux is None else aux
    aligned = loans[[C.ID]].merge(aux, on=C.ID, how="left", validate="1:1")
    if aligned["funded_amnt"].isna().any():
        raise ValueError("some loans have no aux row")
    return aligned


def save_model(
    pipeline, policy: dict, model_path: Path = MODEL_PATH, policy_path: Path = POLICY_PATH
):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")


def load_model(model_path: Path = MODEL_PATH, policy_path: Path = POLICY_PATH):
    return joblib.load(model_path), json.loads(policy_path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
