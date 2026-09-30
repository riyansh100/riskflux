"""The simulated clock: windows derived from as_of, and the params.yaml rewrite."""

import shutil
from pathlib import Path

import pytest

from riskflux.config import load_params, windows_for
from riskflux.retrain.plan import plan

REPO_PARAMS = Path(__file__).parents[2] / "params.yaml"


@pytest.fixture
def future_params(tmp_path) -> Path:
    """A copy of params.yaml in a world where outcomes exist up to 2021-09."""
    path = tmp_path / "params.yaml"
    shutil.copy(REPO_PARAMS, path)
    text = path.read_text()
    text = text.replace('data_snapshot: "2020-09"', 'data_snapshot: "2021-09"')
    text = text.replace('replay: {start: "2017-05"', 'replay: {start: "2018-07"')
    path.write_text(text)
    return path


def test_current_clock_reproduces_the_committed_windows():
    params = load_params(REPO_PARAMS)
    population, splits = windows_for(params.retraining.as_of, params.retraining)
    assert population == params.population.window and splits == params.splits


def test_same_clock_means_nothing_to_retrain():
    assert plan("2020-09", REPO_PARAMS) == []


def test_clock_stops_at_the_data_snapshot():
    assert plan("2026-09", REPO_PARAMS) == []  # no outcomes exist after 2020-09


def test_clock_cannot_go_backwards():
    with pytest.raises(ValueError, match="backwards"):
        plan("2019-09", REPO_PARAMS)


def test_moving_the_clock_forward_shifts_every_window(future_params):
    changes = plan("2021-03", future_params, apply=True)
    assert "population end 2017-04 -> 2017-10" in changes
    params = load_params(future_params)  # rewritten file passes every config check
    assert f"{params.splits['test'].start:%Y-%m}..{params.splits['test'].end:%Y-%m}" == (
        "2017-01..2017-10"
    )
    text = future_params.read_text()
    assert "# derived from clock.as_of" in text  # comments survive the rewrite
    assert 'test: {start: "2017-01", end: "2017-10"}' in text  # dates stay quoted strings
    assert plan("2021-03", future_params) == []  # applying again is a no-op
