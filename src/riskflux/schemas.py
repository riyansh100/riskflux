"""Data contracts, all generated from the column registry and one set of bounds.

- CLEAN_SCHEMA / AUX_SCHEMA (pandera): batch data written by the clean stage.
- LoanApplication (pydantic): one loan arriving at the API.
- applications_to_frame: turns validated requests into the exact frame shape the
  clean stage produces, so the shared feature pipeline sees identical input.

strict=True / extra="forbid": an unexpected column or field is rejected, a second
guard against leakage columns (e.g. grade, int_rate) reaching the model.
"""

from collections.abc import Sequence
from datetime import date
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from riskflux import columns as C

# Bounds beyond ">= 0". Everything is inclusive.
BOUNDS = {
    "loan_amnt": (500, 40_000),  # LC's lending limits
    "emp_length": (0, 10),
    "dti": (0, 100),
    "fico_range_low": (300, 850),
    "fico_range_high": (300, 850),
    "revol_util": (0, 1_000),  # >100% is possible (over the limit); EDA max is 892%
    "bc_util": (0, 1_000),
    "pct_tl_nvr_dlq": (0, 100),
    "percent_bc_gt_75": (0, 100),
}
POSITIVE = {"annual_inc"}  # strictly > 0: used as a divisor
# Core application fields: missing means a broken record, not "unknown".
REQUIRED = {"loan_amnt", "annual_inc", "fico_range_low", "fico_range_high"}

HOME_OWNERSHIP = ["MORTGAGE", "RENT", "OWN", "ANY", "OTHER", "NONE"]
PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "medical", "car", "small_business", "vacation", "moving", "house",
    "renewable_energy", "wedding", "educational",
]  # fmt: skip
ALLOWED = {"home_ownership": HOME_OWNERSHIP, "purpose": PURPOSE}


# --- batch contracts (pandera) ------------------------------------------------------
def _numeric_check(name: str) -> pa.Check:
    if name in POSITIVE:
        return pa.Check.gt(0)
    low, high = BOUNDS.get(name, (0, None))
    return pa.Check.in_range(low, high) if high is not None else pa.Check.ge(low)


def _loan_schema(outcome_known: bool) -> pa.DataFrameSchema:
    """Typed loan data. outcome_known=False is for replayed (not yet matured) loans:
    the label may be missing and the default-rate sanity check doesn't apply."""
    label = (
        pa.Column("int8", pa.Check.isin([0, 1]))
        if outcome_known
        else pa.Column("Int8", pa.Check.isin([0, 1]), nullable=True)
    )
    checks = [
        pa.Check(
            lambda df: df["fico_range_high"] >= df["fico_range_low"], name="fico_range_ordered"
        ),
        pa.Check(
            lambda df: df["earliest_cr_line"] <= df[C.ISSUE_DATE], name="credit_line_before_issue"
        ),
    ]
    if outcome_known:
        # Catches label bugs (e.g. everything mapped to 0). EDA: 14.7%.
        checks.append(
            pa.Check(lambda df: 0.05 <= df[C.TARGET].mean() <= 0.35, name="default_rate_plausible")
        )
    return pa.DataFrameSchema(
        {
            C.ID: pa.Column(int, pa.Check.ge(0), unique=True),
            C.ISSUE_DATE: pa.Column("datetime64[ns]"),
            C.TARGET: label,
            **{
                name: pa.Column(float, _numeric_check(name), nullable=name not in REQUIRED)
                for name in C.NUMERIC_FEATURES
            },
            **{
                name: pa.Column(str, pa.Check.isin(ALLOWED[name]))
                for name in C.CATEGORICAL_FEATURES
            },
            **{name: pa.Column("datetime64[ns]") for name in C.DATE_FEATURES},
        },
        checks=checks,
        strict=True,
        ordered=True,
    )


CLEAN_SCHEMA = _loan_schema(outcome_known=True)
REPLAY_SCHEMA = _loan_schema(outcome_known=False)

AUX_SCHEMA = pa.DataFrameSchema(
    {
        C.ID: pa.Column(int, pa.Check.ge(0), unique=True),
        **{name: pa.Column(float, pa.Check.ge(0), nullable=True) for name in C.AUX_NUMERIC},
        **{name: pa.Column(str, nullable=True) for name in C.AUX_CATEGORICAL},
    },
    strict=True,
    ordered=True,
)


# --- serving contract (pydantic) ----------------------------------------------------
class _ApplicationRules(BaseModel):
    """Cross-field rules, mirroring CLEAN_SCHEMA's dataframe-level checks."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _cross_field_rules(self):
        if self.fico_range_high < self.fico_range_low:
            raise ValueError("fico_range_high must be >= fico_range_low")
        if self.earliest_cr_line > self.application_date:
            raise ValueError("earliest_cr_line must not be after application_date")
        return self


def _numeric_field(name: str) -> tuple:
    if name in POSITIVE:
        limits = {"gt": 0}
    else:
        low, high = BOUNDS.get(name, (0, None))
        limits = {"ge": low} | ({"le": high} if high is not None else {})
    if name in REQUIRED:
        return (float, Field(**limits))
    return (float | None, Field(default=None, **limits))


LoanApplication = create_model(
    "LoanApplication",
    __base__=_ApplicationRules,
    # The decision date plays the role of issue_d. Defaults to today.
    application_date=(date, Field(default_factory=date.today)),
    **{name: _numeric_field(name) for name in C.NUMERIC_FEATURES},
    **{name: (Literal[tuple(ALLOWED[name])], ...) for name in C.CATEGORICAL_FEATURES},
    **{name: (date, ...) for name in C.DATE_FEATURES},
)


def _month_start(values: pd.Series) -> pd.Series:
    """LC records dates at month granularity; align serving dates the same way."""
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


def applications_to_frame(applications: Sequence[BaseModel]) -> pd.DataFrame:
    """Validated requests -> frame with the same columns and dtypes the clean stage
    produces for FEATURES (+ issue_d), i.e. what the feature pipeline expects."""
    records = pd.DataFrame.from_records([a.model_dump() for a in applications])
    frame = pd.DataFrame(index=records.index)
    frame[C.ISSUE_DATE] = _month_start(records["application_date"])
    for name in C.NUMERIC_FEATURES:
        frame[name] = records[name].astype("float64")  # None -> NaN
    for name in C.CATEGORICAL_FEATURES:
        frame[name] = records[name].astype("str")
    for name in C.DATE_FEATURES:
        frame[name] = _month_start(records[name])
    return frame
