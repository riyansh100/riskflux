"""Shared fixtures: synthetic raw rows (ingest-shaped) and random clean-shaped loans."""

import numpy as np
import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.config import PopulationParams, Window
from riskflux.schemas import BOUNDS, CLEAN_SCHEMA, PURPOSE, REQUIRED

# One valid, in-population raw loan. Tests override single fields to probe one rule at a time.
VALID_RAW_ROW = {
    **{col: "1" for col in C.RAW_COLUMNS},
    C.ID: "1001",
    C.ISSUE_DATE: "Mar-2014",
    C.TERM: " 36 months",
    C.APPLICATION_TYPE: "Individual",
    C.LOAN_STATUS: "Fully Paid",
    "loan_amnt": "10000",
    "emp_length": "5 years",
    "annual_inc": "60000",
    "dti": "18.5",
    "fico_range_low": "700",
    "fico_range_high": "704",
    "revol_util": "52.3%",
    "home_ownership": "RENT",
    "purpose": "credit_card",
    "earliest_cr_line": "Jan-2000",
    "int_rate": " 11.53%",
    "installment": "330.0",
    "funded_amnt": "10000",
    "grade": "B",
    "sub_grade": "B3",
    "initial_list_status": "w",
    "verification_status": "Verified",
    "addr_state": "NY",
    "zip_code": "100xx",
}


def make_raw(*rows: dict) -> pd.DataFrame:
    """Build a raw frame; each row is VALID_RAW_ROW with the given overrides.
    Unique ids are assigned unless a row sets one explicitly."""
    records = [
        {**VALID_RAW_ROW, C.ID: str(1000 + i), **overrides} for i, overrides in enumerate(rows)
    ]
    return pd.DataFrame(records, columns=C.RAW_COLUMNS).astype("str")


@pytest.fixture
def population() -> PopulationParams:
    return PopulationParams(
        term_months=36,
        application_type="Individual",
        window=Window.from_months("2013-01", "2017-04"),
    )


def make_clean_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random loans shaped exactly like the clean stage output (passes CLEAN_SCHEMA).
    Enough rows to fit a tiny model, for tests that need predictions without real data."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2013-01-01", "2017-04-01", freq="MS")
    issue = pd.Series(rng.choice(months, n)).astype("datetime64[ns]")

    frame = pd.DataFrame({C.ID: np.arange(n, dtype="int64"), C.ISSUE_DATE: issue})
    frame[C.TARGET] = (rng.random(n) < 0.15).astype("int8")
    for name in C.NUMERIC_FEATURES:
        low, high = BOUNDS.get(name, (0, 100))
        values = rng.uniform(low, high, n).round(1)
        if name not in REQUIRED:
            values[rng.random(n) < 0.1] = np.nan  # some missing, like real data
        frame[name] = values
    frame["annual_inc"] = rng.uniform(20_000, 200_000, n).round(0)
    frame["fico_range_low"] = rng.integers(132, 160, n) * 5.0  # 660 .. 795
    frame["fico_range_high"] = frame["fico_range_low"] + 4
    frame["home_ownership"] = pd.Series(rng.choice(["RENT", "OWN", "MORTGAGE"], n), dtype="str")
    frame["purpose"] = pd.Series(rng.choice(PURPOSE[:5], n), dtype="str")
    history = rng.integers(36, 400, n)
    frame["earliest_cr_line"] = pd.Series(
        [(d.to_period("M") - int(k)).to_timestamp() for d, k in zip(issue, history, strict=True)]
    ).astype("datetime64[ns]")
    return CLEAN_SCHEMA.validate(frame)
