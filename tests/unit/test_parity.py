"""Train/serve parity: the same loan must get identical model input AND an identical
score whether it arrives as a row of the training data or as a JSON request to the API.

    batch path:    clean-stage frame ───────────────────────────────────┐
                                                                       ├─> fitted pipeline
    serving path:  JSON -> LoanApplication -> applications_to_frame ───┘
"""

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.data.clean import clean
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.io import load_model
from riskflux.schemas import LoanApplication, applications_to_frame
from tests.conftest import make_clean_frame, make_raw

ROOT = Path(__file__).parents[2]
TEST_SPLIT = ROOT / "data/processed/test.parquet"
TRAINED_MODEL = ROOT / "models/model.joblib"
TRAINED_POLICY = ROOT / "models/policy.json"


def to_request_json(row: pd.Series) -> str:
    """Serialize a training row the way an API client would send it."""
    body = {"application_date": row[C.ISSUE_DATE].date().isoformat()}
    for name in C.FEATURES:
        value = row[name]
        if isinstance(value, pd.Timestamp):
            body[name] = value.date().isoformat()
        elif isinstance(value, float) and math.isnan(value):
            body[name] = None
        else:
            body[name] = value
    return json.dumps(body)


def serving_frame(loans: pd.DataFrame) -> pd.DataFrame:
    """What the API hands the pipeline after parsing each loan's JSON request."""
    bodies = [to_request_json(row) for _, row in loans.iterrows()]
    return applications_to_frame([LoanApplication.model_validate_json(b) for b in bodies])


def assert_feature_parity(loans: pd.DataFrame) -> None:
    loans = loans.reset_index(drop=True)
    features = build_feature_pipeline().fit(loans)
    serving = features.transform(serving_frame(loans))
    pd.testing.assert_frame_equal(features.transform(loans), serving)


def assert_score_parity(pipeline, loans: pd.DataFrame) -> None:
    loans = loans.reset_index(drop=True)
    batch = pipeline.predict_proba(loans)[:, 1]
    serving = pipeline.predict_proba(serving_frame(loans))[:, 1]
    np.testing.assert_array_equal(batch, serving)  # exact, not approximate


def test_feature_parity_on_edge_case_loans(population):
    loans, _ = clean(
        make_raw(
            {},
            {"emp_length": None, "mths_since_last_delinq": None, "dti": None},  # missing values
            {"home_ownership": "OWN", "purpose": "car", "emp_length": "10+ years"},
            {"earliest_cr_line": "Jan-1970", C.ISSUE_DATE: "Apr-2017"},
        ),
        population,
    )
    assert_feature_parity(loans)


def test_score_parity_with_a_small_trained_model():
    train, holdout = make_clean_frame(400, seed=1), make_clean_frame(200, seed=2)
    features = build_feature_pipeline().fit(train)
    lgbm = lgb.LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1, random_state=0)
    lgbm.fit(features.transform(train), train[C.TARGET])
    calibrated = IsotonicCalibrated(lgbm).fit(features.transform(holdout), holdout[C.TARGET])
    assert_score_parity(Pipeline([*features.steps, ("model", calibrated)]), holdout)


@pytest.mark.skipif(not TEST_SPLIT.exists(), reason="needs data/processed (dvc pull)")
def test_feature_parity_on_real_loans():
    assert_feature_parity(pd.read_parquet(TEST_SPLIT).sample(n=2_000, random_state=0))


@pytest.mark.skipif(
    not (TEST_SPLIT.exists() and TRAINED_MODEL.exists()), reason="needs data + model (dvc pull)"
)
def test_score_parity_of_the_trained_model_on_real_loans():
    pipeline, _ = load_model(TRAINED_MODEL, TRAINED_POLICY)
    assert_score_parity(pipeline, pd.read_parquet(TEST_SPLIT).sample(n=2_000, random_state=0))
