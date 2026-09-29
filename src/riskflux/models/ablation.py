"""Ablation stage (D4): how much signal is in Lending Club's own underwriting outputs?

Compares ranking quality (ROC/PR-AUC) on test for:
  - LC's interest rate alone as a risk score (LC's own model, priced)
  - our model (no LC decision columns)       <- the served model
  - our model + grade, sub_grade, int_rate   <- what we'd get by stacking on LC's model

Analysis only: nothing here is served or feeds back into training. Uncalibrated
LightGBM is enough because AUC depends only on ranking.

Run: uv run python -m riskflux.models.ablation
"""

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from riskflux import columns as C
from riskflux.config import ABLATION_METRICS, load_params
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.io import aux_for, load_split, write_json
from riskflux.models.train import fit_lightgbm

LC_COLUMNS = ["grade", "sub_grade", "int_rate"]


def with_lc_columns(X: pd.DataFrame, aux: pd.DataFrame, categories: dict) -> pd.DataFrame:
    X = X.copy()
    X["int_rate"] = aux["int_rate"].to_numpy()
    for col in ("grade", "sub_grade"):
        X[col] = pd.Categorical(aux[col].to_numpy(), categories=categories[col])
    return X


def ranking(y, score) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, score)), 5),
        "pr_auc": round(float(average_precision_score(y, score)), 5),
    }


def main() -> None:
    params = load_params()
    splits = {name: load_split(name) for name in ("train", "val", "test")}
    aux = {name: aux_for(df) for name, df in splits.items()}
    y = {name: df[C.TARGET].to_numpy() for name, df in splits.items()}

    features = build_feature_pipeline().fit(splits["train"])
    X = {name: features.transform(df) for name, df in splits.items()}
    categories = {c: sorted(aux["train"][c].dropna().unique()) for c in ("grade", "sub_grade")}
    X_lc = {name: with_lc_columns(X[name], aux[name], categories) for name in X}

    ours = fit_lightgbm(X["train"], y["train"], X["val"], y["val"], params.model)
    ours_lc = fit_lightgbm(X_lc["train"], y["train"], X_lc["val"], y["val"], params.model)

    results = {
        "lc_int_rate_only": ranking(y["test"], aux["test"]["int_rate"]),
        "our_model": ranking(y["test"], ours.predict_proba(X["test"])[:, 1]),
        "our_model_plus_lc_grade": ranking(y["test"], ours_lc.predict_proba(X_lc["test"])[:, 1]),
    }
    write_json(ABLATION_METRICS, results)
    for name, r in results.items():
        print(f"{name:25s} ROC-AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}")


if __name__ == "__main__":
    main()
