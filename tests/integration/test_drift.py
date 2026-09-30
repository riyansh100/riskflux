"""The drift job must stay quiet on unchanged data and alert on the shifts it's built for."""

import numpy as np
import pytest

from riskflux.config import MonitoringParams, Window
from riskflux.monitoring.drift import PD, load_prediction_log, monitored, summarize_window
from riskflux.schemas import LoanApplication, applications_to_frame
from riskflux.serving.client import frame_to_requests
from riskflux.serving.logs import PredictionSink
from tests.conftest import make_clean_frame

PARAMS = MonitoringParams(
    replay=Window.from_months("2017-05", "2020-09"),
    sample_per_month=2000,
    seed=0,
    reference_sample=2000,
    psi_threshold=0.1,
    drifted_share_alert=0.3,
    prediction_psi_alert=0.2,
    missing_share_alert=0.1,
)


def scored(frame, seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    out = monitored(frame, np.clip(rng.beta(2, 10, len(frame)) + shift, 0, 1))
    out["decision"] = np.where(out[PD] >= 0.24, "reject", "approve")
    return out


@pytest.fixture(scope="module")
def reference():
    return scored(make_clean_frame(2000, seed=1), seed=1)


def test_no_alert_when_nothing_changed(reference):
    same = scored(make_clean_frame(2000, seed=2), seed=2)
    window = summarize_window(same, reference, PARAMS)
    assert window["alerts"] == []
    assert window["drifted_share"] < PARAMS.drifted_share_alert
    assert window["prediction_psi"] < PARAMS.prediction_psi_alert


def test_alerts_when_inputs_and_predictions_shift(reference):
    frame = make_clean_frame(2000, seed=3)
    for name in ["loan_amnt", "annual_inc", "dti", "revol_bal", "revol_util", "total_acc"]:
        frame[name] = frame[name] * 3  # a big, broad change in who applies
    for name in [c for c in frame.columns if c.startswith(("num_", "mo_sin", "mths_since"))]:
        frame[name] = frame[name] + 50
    shifted = scored(frame, seed=3, shift=0.25)
    window = summarize_window(shifted, reference, PARAMS)
    assert window["drifted_share"] >= PARAMS.drifted_share_alert
    assert window["prediction_psi"] >= PARAMS.prediction_psi_alert
    assert len(window["alerts"]) >= 2


def test_alerts_when_an_input_goes_missing(reference):
    """PSI ignores missing values; the missing-rate rule catches a feed going blank."""
    frame = make_clean_frame(2000, seed=4)
    frame.loc[frame.index[:600], "dti"] = np.nan  # +30 points missing
    window = summarize_window(scored(frame, seed=4), reference, PARAMS)
    assert any("missing rate of dti" in a for a in window["alerts"])


def test_prediction_log_round_trips_through_the_serving_contract(tmp_path):
    loans = make_clean_frame(20, seed=5)
    sink = PredictionSink(tmp_path / "predictions.jsonl")
    for body in frame_to_requests(loans):
        application = LoanApplication.model_validate(body)
        result = {"model_version": "1", "pd": 0.1, "decision": "approve", "policy_flags": []}
        sink.write("req", application.model_dump(mode="json"), result)

    logged = load_prediction_log(tmp_path / "predictions.jsonl")
    expected = applications_to_frame(
        [LoanApplication.model_validate(b) for b in frame_to_requests(loans)]
    )
    assert (logged[PD] == 0.1).all() and len(logged) == 20
    np.testing.assert_array_equal(
        monitored(logged, logged[PD]).drop(columns=PD).to_numpy(dtype=str),
        monitored(expected, logged[PD]).drop(columns=PD).to_numpy(dtype=str),
    )
