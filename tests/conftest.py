"""Shared fixtures: synthetic raw rows shaped exactly like the ingest output (all strings)."""

import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.config import PopulationParams, Window

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
