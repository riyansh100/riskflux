"""Stateless feature engineering, shared by training and serving.

Pure function of one loan's own fields: no statistics from other rows, so it gives
the same answer for a batch of 500k loans and for a single API request.

Input: a typed frame shaped like the clean stage output (or the serving frame built
from a LoanApplication). Output: exactly MODEL_FEATURES, in that order.
"""

import pandas as pd

from riskflux import columns as C

REPLACED = ["fico_range_low", "fico_range_high"]  # always 4 points apart -> midpoint
DERIVED = ["fico_mid", "credit_history_months", "loan_to_income", "revol_bal_to_income"]

MODEL_NUMERIC = [c for c in C.NUMERIC_FEATURES if c not in REPLACED] + DERIVED
MODEL_CATEGORICAL = list(C.CATEGORICAL_FEATURES)
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL

# issue_d is needed to compute credit-history length but is NOT itself a model input:
# as a feature it would let the model learn "when" instead of "how risky".
INPUT_COLUMNS = [C.ISSUE_DATE, *C.FEATURES]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature input is missing columns: {missing}")

    out = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    out["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    issued, opened = df[C.ISSUE_DATE], df["earliest_cr_line"]
    months = (issued.dt.year - opened.dt.year) * 12 + (issued.dt.month - opened.dt.month)
    out["credit_history_months"] = months.astype("float64")

    out["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    out["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"]
    return out[MODEL_FEATURES]
