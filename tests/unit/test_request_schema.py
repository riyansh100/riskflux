"""The API contract (LoanApplication) must reject what the batch contract rejects."""

import pytest
from pydantic import ValidationError

from riskflux import columns as C
from riskflux.schemas import LoanApplication

MINIMAL_REQUEST = {
    "application_date": "2016-03-15",
    "loan_amnt": 10_000,
    "annual_inc": 60_000,
    "fico_range_low": 700,
    "fico_range_high": 704,
    "home_ownership": "RENT",
    "purpose": "credit_card",
    "earliest_cr_line": "2009-11-01",
}


def test_minimal_request_is_valid_and_optional_fields_default_to_missing():
    app = LoanApplication.model_validate(MINIMAL_REQUEST)
    assert app.dti is None
    assert app.mths_since_last_delinq is None


def test_request_fields_are_exactly_the_model_inputs():
    assert set(LoanApplication.model_fields) == {"application_date", *C.FEATURES}


@pytest.mark.parametrize("leaky_field", ["grade", "int_rate", "total_pymnt", "loan_status"])
def test_rejects_leakage_fields(leaky_field):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LoanApplication.model_validate({**MINIMAL_REQUEST, leaky_field: 1})


@pytest.mark.parametrize(
    "override",
    [
        {"fico_range_low": 900},  # out of range
        {"annual_inc": 0},  # must be > 0 (divisor)
        {"dti": -1},  # can't be negative
        {"home_ownership": "CASTLE"},  # unknown category
        {"fico_range_low": 710, "fico_range_high": 704},  # range inverted
        {"earliest_cr_line": "2020-01-01"},  # credit line opened after application
    ],
)
def test_rejects_invalid_values(override):
    with pytest.raises(ValidationError):
        LoanApplication.model_validate({**MINIMAL_REQUEST, **override})


@pytest.mark.parametrize("required", ["loan_amnt", "annual_inc", "purpose", "earliest_cr_line"])
def test_rejects_missing_required_field(required):
    request = {k: v for k, v in MINIMAL_REQUEST.items() if k != required}
    with pytest.raises(ValidationError):
        LoanApplication.model_validate(request)
