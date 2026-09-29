"""Data contracts (pandera) for the clean and aux datasets.

strict=True: an unexpected column fails validation, which is a second guard against
leakage columns ending up in the model's input file.
"""

import pandera.pandas as pa

from riskflux import columns as C

NON_NEGATIVE = pa.Check.ge(0)

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
# Core application fields: missing means a broken record, not "unknown".
REQUIRED = {"loan_amnt", "annual_inc", "fico_range_low", "fico_range_high"}

HOME_OWNERSHIP = ["MORTGAGE", "RENT", "OWN", "ANY", "OTHER", "NONE"]
PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "medical", "car", "small_business", "vacation", "moving", "house",
    "renewable_energy", "wedding", "educational",
]  # fmt: skip


def _numeric_column(name: str) -> pa.Column:
    low, high = BOUNDS.get(name, (0, None))
    check = pa.Check.in_range(low, high) if high is not None else pa.Check.ge(low)
    return pa.Column(float, check, nullable=name not in REQUIRED)


CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        C.ID: pa.Column(int, NON_NEGATIVE, unique=True),
        C.ISSUE_DATE: pa.Column("datetime64[ns]"),
        C.TARGET: pa.Column("int8", pa.Check.isin([0, 1])),
        **{name: _numeric_column(name) for name in C.NUMERIC_FEATURES},
        "annual_inc": pa.Column(float, pa.Check.gt(0)),
        "home_ownership": pa.Column(str, pa.Check.isin(HOME_OWNERSHIP)),
        "purpose": pa.Column(str, pa.Check.isin(PURPOSE)),
        "earliest_cr_line": pa.Column("datetime64[ns]"),
    },
    checks=[
        pa.Check(
            lambda df: df["fico_range_high"] >= df["fico_range_low"], name="fico_range_ordered"
        ),
        pa.Check(
            lambda df: df["earliest_cr_line"] <= df[C.ISSUE_DATE], name="credit_line_before_issue"
        ),
        # Catches label bugs (e.g. everything mapped to 0). EDA: 14.7%.
        pa.Check(lambda df: 0.05 <= df[C.TARGET].mean() <= 0.35, name="default_rate_plausible"),
    ],
    strict=True,
    ordered=True,
)

AUX_SCHEMA = pa.DataFrameSchema(
    {
        C.ID: pa.Column(int, NON_NEGATIVE, unique=True),
        **{name: pa.Column(float, NON_NEGATIVE, nullable=True) for name in C.AUX_NUMERIC},
        **{name: pa.Column(str, nullable=True) for name in C.AUX_CATEGORICAL},
    },
    strict=True,
    ordered=True,
)
