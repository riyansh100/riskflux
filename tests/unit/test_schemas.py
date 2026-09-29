"""The data contract must reject the failure modes it exists to catch."""

import pandas as pd
import pandera.pandas as pa
import pytest

from riskflux import columns as C
from riskflux.data.clean import clean
from riskflux.schemas import CLEAN_SCHEMA
from tests.conftest import make_raw

# pandera raises SchemaErrors (plural) for column-set problems, SchemaError for row checks.
INVALID = (pa.errors.SchemaError, pa.errors.SchemaErrors)


@pytest.fixture
def valid(population) -> pd.DataFrame:
    statuses = ["Fully Paid"] * 8 + ["Charged Off"] * 2
    loans, _ = clean(make_raw(*({C.LOAN_STATUS: s} for s in statuses)), population)
    return loans


def test_valid_data_passes(valid):
    CLEAN_SCHEMA.validate(valid)


def test_rejects_unexpected_column(valid):
    """strict=True: a leakage column sneaking into the model file fails validation."""
    with pytest.raises(INVALID):
        CLEAN_SCHEMA.validate(valid.assign(total_pymnt=1.0))


def test_rejects_duplicate_ids(valid):
    with pytest.raises(INVALID):
        CLEAN_SCHEMA.validate(valid.assign(**{C.ID: 1}))


def test_rejects_invalid_label(valid):
    bad = valid.copy()
    bad.loc[0, C.TARGET] = 2
    with pytest.raises(INVALID):
        CLEAN_SCHEMA.validate(bad)


def test_rejects_implausible_default_rate(valid):
    all_good = pd.Series(0, index=valid.index, dtype="int8")
    with pytest.raises(INVALID, match="default_rate_plausible"):
        CLEAN_SCHEMA.validate(valid.assign(**{C.TARGET: all_good}))


def test_rejects_credit_line_opened_after_issue(valid):
    bad = valid.copy()
    bad.loc[0, "earliest_cr_line"] = pd.Timestamp("2020-01-01")
    with pytest.raises(INVALID, match="credit_line_before_issue"):
        CLEAN_SCHEMA.validate(bad)


def test_rejects_out_of_range_fico(valid):
    bad = valid.copy()
    bad.loc[0, "fico_range_low"] = 900.0
    with pytest.raises(INVALID):
        CLEAN_SCHEMA.validate(bad)
