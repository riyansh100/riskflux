import pickle

import lightgbm as lgb
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.metrics import calibration_table, expected_calibration_error
from tests.conftest import make_clean_frame

TINY_LGBM = {"n_estimators": 20, "min_child_samples": 5, "verbose": -1, "random_state": 0}


@pytest.fixture(scope="module")
def fitted():
    train, holdout = make_clean_frame(400, seed=1), make_clean_frame(400, seed=2)
    features = build_feature_pipeline().fit(train)
    lgbm = lgb.LGBMClassifier(**TINY_LGBM).fit(features.transform(train), train[C.TARGET])
    calibrated = IsotonicCalibrated(lgbm).fit(features.transform(holdout), holdout[C.TARGET])
    return Pipeline([*features.steps, ("model", calibrated)]), holdout


def test_probabilities_are_valid_and_rank_preserving(fitted):
    pipeline, holdout = fitted
    raw = pipeline[-1].predict_raw(pipeline[:-1].transform(holdout))
    proba = pipeline.predict_proba(holdout)
    assert proba.shape == (len(holdout), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()
    # Isotonic is monotone: a higher raw score never gets a lower calibrated PD.
    order = np.argsort(raw)
    assert (np.diff(proba[order, 1]) >= -1e-12).all()


def test_calibrated_pipeline_survives_pickling(fitted):
    pipeline, holdout = fitted
    reloaded = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(pipeline.predict_proba(holdout), reloaded.predict_proba(holdout))


def test_unfitted_calibrator_is_reported_as_unfitted():
    assert not IsotonicCalibrated(model=None).__sklearn_is_fitted__()


def test_ece_is_zero_for_perfect_calibration_and_large_when_off():
    rng = np.random.default_rng(0)
    pd_hat = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, pd_hat.size) < pd_hat).astype(int)
    assert expected_calibration_error(y, pd_hat) < 0.01
    assert expected_calibration_error(y, np.clip(pd_hat + 0.2, 0, 1)) > 0.15


def test_calibration_table_has_equal_size_bins():
    table = calibration_table(np.zeros(1_000), np.linspace(0, 1, 1_000), bins=10)
    assert table["loans"].tolist() == [100] * 10
