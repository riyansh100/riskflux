"""End-to-end registry flow against a throwaway local MLflow store (no network):
log -> @challenger -> promote -> @champion, plus the guardrails."""

import json

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.config import CHALLENGER, CHAMPION, REGISTERED_MODEL, PromotionParams
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.io import save_model
from riskflux.registry.log_run import register_challenger
from riskflux.registry.promote import promote
from tests.conftest import make_clean_frame

RULES = PromotionParams(min_cost_improvement=0.001, max_auc_drop=0.005, max_ece=0.5)
POLICY = {"threshold": 0.2, "cost_model": {"lgd": 0.5, "haircut": 0.8, "term_months": 36}}


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    # MLflow 3 disables the file store by default; fine for an isolated test store.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    return mlflow.MlflowClient()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    train, holdout = make_clean_frame(400, seed=1), make_clean_frame(300, seed=2)
    features = build_feature_pipeline().fit(train)
    lgbm = lgb.LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1, random_state=0)
    lgbm.fit(features.transform(train), train[C.TARGET])
    calibrated = IsotonicCalibrated(lgbm).fit(features.transform(holdout), holdout[C.TARGET])
    pipeline = Pipeline([*features.steps, ("model", calibrated)])

    out = tmp_path_factory.mktemp("model")
    model_path, policy_path = out / "model.joblib", out / "policy.json"
    save_model(pipeline, POLICY, model_path, policy_path)
    aux = pd.DataFrame(
        {
            C.ID: holdout[C.ID],
            "funded_amnt": holdout["loan_amnt"],
            "installment": holdout["loan_amnt"] * 0.035,  # ~10% APR over 36 months
        }
    )
    return pipeline, model_path, policy_path, holdout, aux


def register(trained, **tags) -> str:
    _, model_path, policy_path, loans, _ = trained
    example = loans.head(5)[[C.ISSUE_DATE, *C.FEATURES]]
    all_tags = {"git_commit": "abc1234", "git_dirty": "false"} | tags
    return register_challenger(model_path, policy_path, example, {"p": 1}, {"m": 1.0}, all_tags)


def alias(client, name: str) -> str:
    return client.get_model_version_by_alias(REGISTERED_MODEL, name).version


def test_registered_model_scores_exactly_like_the_pipeline(local_mlflow, trained):
    pipeline, *_, loans, _ = trained
    register(trained)
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@{CHALLENGER}")
    scored = loaded.predict(loans[[C.ISSUE_DATE, *C.FEATURES]])
    expected = pipeline.predict_proba(loans)[:, 1]
    np.testing.assert_array_equal(scored["pd"].to_numpy(), expected)
    np.testing.assert_array_equal(scored["reject"].to_numpy(), expected >= POLICY["threshold"])


def test_champion_challenger_flow(local_mlflow, trained):
    *_, loans, aux = trained

    first = register(trained)
    assert alias(local_mlflow, CHALLENGER) == first
    promoted, reason = promote(loans, aux, RULES)
    assert promoted and "bootstrap" in reason
    assert alias(local_mlflow, CHAMPION) == first

    # Same model again: no cost improvement -> champion stays put.
    second = register(trained)
    assert alias(local_mlflow, CHALLENGER) == second
    promoted, reason = promote(loans, aux, RULES)
    assert not promoted and "improvement" in reason
    assert alias(local_mlflow, CHAMPION) == first
    decision = local_mlflow.get_model_version(REGISTERED_MODEL, second).tags["promotion_decision"]
    assert "improvement" in decision


def test_version_logged_from_dirty_code_is_never_promoted(local_mlflow, trained):
    *_, loans, aux = trained
    register(trained, git_dirty="true")
    promoted, reason = promote(loans, aux, RULES)
    assert not promoted and "uncommitted" in reason


def test_policy_is_logged_with_the_run(local_mlflow, trained):
    register(trained)
    run_id = local_mlflow.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER).run_id
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="policy.json")
    assert json.loads(open(local).read())["threshold"] == POLICY["threshold"]
