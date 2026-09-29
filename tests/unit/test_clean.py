import math

import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.data.clean import clean
from riskflux.schemas import AUX_SCHEMA, CLEAN_SCHEMA
from tests.conftest import make_raw


def test_label_mapping(population):
    raw = make_raw(
        {C.LOAN_STATUS: "Fully Paid"},
        {C.LOAN_STATUS: "Charged Off"},
        {C.LOAN_STATUS: "Default"},
    )
    loans, _ = clean(raw, population)
    assert loans[C.TARGET].tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    "override",
    [
        {C.LOAN_STATUS: "Current"},  # outcome unknown (in flight)
        {C.LOAN_STATUS: "Late (31-120 days)"},
        {C.TERM: " 60 months"},
        {C.APPLICATION_TYPE: "Joint App"},
        {C.ISSUE_DATE: "Dec-2012"},  # before the window
        {C.ISSUE_DATE: "May-2017"},  # after the window (immature cohort)
        {C.ID: "Loans that do not meet the credit policy"},  # junk separator row
    ],
)
def test_rows_outside_population_are_dropped(population, override):
    loans, aux = clean(make_raw({}, override), population)
    assert len(loans) == len(aux) == 1


def test_window_edges_are_inclusive(population):
    raw = make_raw({C.ISSUE_DATE: "Jan-2013"}, {C.ISSUE_DATE: "Apr-2017"})
    loans, _ = clean(raw, population)
    assert len(loans) == 2


def test_parsing(population):
    loans, aux = clean(make_raw({}), population)
    row, aux_row = loans.iloc[0], aux.iloc[0]
    assert row[C.ID] == 1000
    assert row[C.ISSUE_DATE] == pd.Timestamp("2014-03-01")
    assert row["earliest_cr_line"] == pd.Timestamp("2000-01-01")
    assert row["revol_util"] == 52.3  # "52.3%"
    assert row["emp_length"] == 5.0  # "5 years"
    assert aux_row["int_rate"] == 11.53  # " 11.53%"


@pytest.mark.parametrize(
    ("raw_value", "years"), [("< 1 year", 0.0), ("1 year", 1.0), ("10+ years", 10.0)]
)
def test_emp_length_edge_values(population, raw_value, years):
    loans, _ = clean(make_raw({"emp_length": raw_value}), population)
    assert loans["emp_length"].iloc[0] == years


def test_missing_values_stay_missing(population):
    loans, _ = clean(make_raw({"emp_length": None, "mths_since_last_delinq": None}), population)
    assert math.isnan(loans["emp_length"].iloc[0])
    assert math.isnan(loans["mths_since_last_delinq"].iloc[0])


def test_negative_dti_becomes_missing(population):
    loans, _ = clean(make_raw({"dti": "-1"}), population)
    assert math.isnan(loans["dti"].iloc[0])


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("annual_inc", "sixty thousand"), ("emp_length", "n/a"), ("earliest_cr_line", "2000-01")],
)
def test_unparseable_values_fail_loudly(population, column, bad_value):
    with pytest.raises(ValueError, match=column):
        clean(make_raw({column: bad_value}), population)


def test_outputs_have_exactly_the_declared_columns(population):
    loans, aux = clean(make_raw({}), population)
    assert list(loans.columns) == [C.ID, C.ISSUE_DATE, C.TARGET, *C.FEATURES]
    assert list(aux.columns) == [C.ID, *C.AUX]


def test_outputs_satisfy_the_data_contracts(population):
    # Mix of labels so the default-rate check has something plausible to evaluate.
    statuses = ["Fully Paid"] * 8 + ["Charged Off"] * 2
    loans, aux = clean(make_raw(*({C.LOAN_STATUS: s} for s in statuses)), population)
    CLEAN_SCHEMA.validate(loans)
    AUX_SCHEMA.validate(aux)
