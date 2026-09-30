import pandas as pd

from riskflux import columns as C
from riskflux.config import PopulationParams, Window
from riskflux.data.clean import clean
from riskflux.data.replay import sample_by_month
from riskflux.schemas import REPLAY_SCHEMA
from tests.conftest import make_raw

REPLAY_POPULATION = PopulationParams(
    term_months=36, application_type="Individual", window=Window.from_months("2017-05", "2020-09")
)


def test_replay_keeps_in_flight_loans_with_missing_label():
    raw = make_raw(
        {C.ISSUE_DATE: "Jun-2019", C.LOAN_STATUS: "Current"},
        {C.ISSUE_DATE: "Jun-2019", C.LOAN_STATUS: "Late (31-120 days)"},
        {C.ISSUE_DATE: "Jun-2018", C.LOAN_STATUS: "Charged Off"},
        {C.ISSUE_DATE: "Jun-2018", C.LOAN_STATUS: "Fully Paid"},
        {C.ISSUE_DATE: "Jun-2016", C.LOAN_STATUS: "Current"},  # before the replay window
    )
    loans, _ = clean(raw, REPLAY_POPULATION, require_outcome=False)
    assert len(loans) == 4
    assert loans[C.TARGET].tolist()[2:] == [1, 0]
    assert loans[C.TARGET].isna().tolist()[:2] == [True, True]
    REPLAY_SCHEMA.validate(loans)


def test_sample_by_month_caps_each_month_deterministically():
    loans = pd.DataFrame(
        {
            C.ID: range(30),
            C.ISSUE_DATE: pd.to_datetime(["2018-01-01"] * 20 + ["2018-02-01"] * 10),
        }
    )
    first, second = sample_by_month(loans, 5, seed=1), sample_by_month(loans, 5, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(C.ISSUE_DATE).size().tolist() == [5, 5]
    assert first[C.ISSUE_DATE].is_monotonic_increasing
    assert len(sample_by_month(loans, 50, seed=1)) == 30  # small months are kept whole
