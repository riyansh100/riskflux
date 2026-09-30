"""API contract tests, including train/serve parity through real HTTP requests."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.io import save_model
from riskflux.serving.app import MAX_BATCH, create_app
from riskflux.serving.client import frame_to_requests
from tests.conftest import make_clean_frame

ROOT = Path(__file__).parents[2]
POLICY = {"threshold": 0.2, "cost_model": {"lgd": 0.5, "haircut": 0.8, "term_months": 36}}
APPLICATION = {
    "application_date": "2016-03-15",
    "loan_amnt": 10_000,
    "annual_inc": 60_000,
    "fico_range_low": 700,
    "fico_range_high": 704,
    "home_ownership": "RENT",
    "purpose": "credit_card",
    "earliest_cr_line": "2009-11-01",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    train, holdout = make_clean_frame(400, seed=1), make_clean_frame(300, seed=2)
    features = build_feature_pipeline().fit(train)
    lgbm = lgb.LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1, random_state=0)
    lgbm.fit(features.transform(train), train[C.TARGET])
    calibrated = IsotonicCalibrated(lgbm).fit(features.transform(holdout), holdout[C.TARGET])
    pipeline = Pipeline([*features.steps, ("model", calibrated)])

    model_dir = tmp_path_factory.mktemp("model")
    save_model(pipeline, POLICY, model_dir / "model.joblib", model_dir / "policy.json")
    (model_dir / "metadata.json").write_text(json.dumps({"version": "7", "alias": "champion"}))
    return pipeline, model_dir, holdout


@pytest.fixture
def client(trained, tmp_path):
    _, model_dir, _ = trained
    app = create_app(model_dir, prediction_log=tmp_path / "predictions.jsonl")
    with TestClient(app) as c:
        c.log_path = tmp_path / "predictions.jsonl"
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "model_version": "7"}


def test_not_ready_without_a_model_but_still_alive(tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as c:
        assert c.get("/health").status_code == 200  # liveness: don't restart-loop
        assert c.get("/ready").status_code == 503  # readiness: don't send traffic
        assert c.post("/predict", json=APPLICATION).status_code == 503


def test_predict_response(client):
    r = client.post("/predict", json=APPLICATION, headers={"x-request-id": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert r.headers["x-request-id"] == body["request_id"] == "abc123"
    assert 0 <= body["pd"] <= 1
    assert body["decision"] == ("reject" if body["pd"] >= POLICY["threshold"] else "approve")
    assert body["expected_loss"] == pytest.approx(body["pd"] * 0.5 * 10_000, abs=0.01)
    assert body["model_version"] == "7" and body["policy_flags"] == []
    contributions = [f["contribution"] for f in body["risk_factors"]]
    assert len(contributions) <= 3 and all(c > 0 for c in contributions)
    assert contributions == sorted(contributions, reverse=True)


@pytest.mark.parametrize(
    ("override", "flag"),
    [
        ({"fico_range_low": 620, "fico_range_high": 624}, "fico_below_660"),
        ({"earliest_cr_line": "2015-01-01"}, "credit_history_under_36_months"),
        ({"loan_amnt": 35_000, "annual_inc": 40_000}, "loan_to_income_above_0.5"),
    ],
)
def test_outside_training_range_is_referred_to_a_human(client, override, flag):
    body = client.post("/predict", json={**APPLICATION, **override}).json()
    assert body["decision"] == "refer" and flag in body["policy_flags"]


@pytest.mark.parametrize(
    "bad", [{"grade": "B"}, {"fico_range_low": 900}, {"purpose": "yacht"}, {"annual_inc": 0}]
)
def test_invalid_requests_are_rejected_with_422(client, bad):
    assert client.post("/predict", json={**APPLICATION, **bad}).status_code == 422


def test_batch_endpoint_and_limit(client):
    r = client.post("/predict/batch", json={"applications": [APPLICATION] * 3})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 3
    too_many = {"applications": [APPLICATION] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_every_prediction_is_logged_for_drift_monitoring(client):
    client.post("/predict/batch", json={"applications": [APPLICATION] * 2})
    records = [json.loads(line) for line in client.log_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["features"]["fico_range_low"] == 700
    assert {"ts", "request_id", "model_version", "pd", "decision"} <= records[0].keys()


def test_api_scores_match_batch_scores_exactly(client, trained):
    """Parity end to end: HTTP JSON -> API == pipeline on the training-format frame."""
    pipeline, _, loans = trained
    bodies = frame_to_requests(loans.head(50))
    api = [
        p["pd"]
        for p in client.post("/predict/batch", json={"applications": bodies}).json()["predictions"]
    ]
    np.testing.assert_array_equal(api, pipeline.predict_proba(loans.head(50))[:, 1])


@pytest.mark.skipif(
    not (ROOT / "models/model.joblib").exists()
    or not (ROOT / "data/processed/test.parquet").exists(),
    reason="needs the trained model and data (dvc pull)",
)
def test_real_model_api_parity_on_real_loans():
    from riskflux.models.io import load_model

    pipeline, _ = load_model(ROOT / "models/model.joblib", ROOT / "models/policy.json")
    loans = pd.read_parquet(ROOT / "data/processed/test.parquet").sample(500, random_state=1)
    bodies = frame_to_requests(loans)
    with TestClient(create_app(ROOT / "models")) as c:
        preds = c.post("/predict/batch", json={"applications": bodies}).json()["predictions"]
    np.testing.assert_array_equal([p["pd"] for p in preds], pipeline.predict_proba(loans)[:, 1])
