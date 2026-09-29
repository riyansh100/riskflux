from pathlib import Path

import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.config import Window, check_splits, load_params
from riskflux.data.split import split

REPO_PARAMS = Path(__file__).parents[2] / "params.yaml"
WINDOWS = {
    "train": Window.from_months("2013-01", "2015-12"),
    "val": Window.from_months("2016-01", "2016-06"),
    "test": Window.from_months("2016-07", "2017-04"),
}


def loans_issued(*months: str) -> pd.DataFrame:
    return pd.DataFrame({C.ID: range(len(months)), C.ISSUE_DATE: pd.to_datetime(list(months))})


def test_each_loan_lands_in_its_time_window():
    parts = split(loans_issued("2013-01-01", "2015-12-01", "2016-01-01", "2017-04-01"), WINDOWS)
    assert parts["train"][C.ID].tolist() == [0, 1]  # both edges of train are inclusive
    assert parts["val"][C.ID].tolist() == [2]
    assert parts["test"][C.ID].tolist() == [3]


def test_splits_are_strictly_ordered_in_time():
    months = pd.date_range("2013-01-01", "2017-04-01", freq="MS").astype(str)
    parts = split(loans_issued(*months), WINDOWS)
    assert parts["train"][C.ISSUE_DATE].max() < parts["val"][C.ISSUE_DATE].min()
    assert parts["val"][C.ISSUE_DATE].max() < parts["test"][C.ISSUE_DATE].min()


def test_loan_outside_all_windows_fails():
    with pytest.raises(ValueError, match="assigned"):
        split(loans_issued("2013-01-01", "2018-01-01"), WINDOWS)


def test_repo_params_are_valid():
    params = load_params(REPO_PARAMS)
    assert list(params.splits) == ["train", "val", "test"]


@pytest.mark.parametrize(
    "windows",
    [
        {**WINDOWS, "val": Window.from_months("2016-02", "2016-06")},  # gap after train
        {**WINDOWS, "val": Window.from_months("2015-12", "2016-06")},  # overlaps train
        {"val": WINDOWS["val"], "train": WINDOWS["train"], "test": WINDOWS["test"]},  # misordered
        {**WINDOWS, "test": Window.from_months("2016-07", "2017-03")},  # misses population end
    ],
)
def test_invalid_split_configs_are_rejected(windows):
    with pytest.raises(ValueError):
        check_splits(Window.from_months("2013-01", "2017-04"), windows)
