"""Isotonic calibration of a fitted classifier on held-out predictions.

Why calibrate: the cost function multiplies PD by dollars, so a predicted 0.20 must
mean "20 of 100 such loans default", not just "riskier than 0.15".

Why our own ~20-line wrapper instead of sklearn's CalibratedClassifierCV: sklearn's
input validation can convert the DataFrame to a plain array, which drops the pandas
category dtype LightGBM relies on. This wrapper passes the frame through untouched.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrated(ClassifierMixin, BaseEstimator):
    """A fitted classifier followed by a monotone map from raw score to probability.

    `model` must already be trained. `fit(X, y)` fits only the calibrator, and X, y must
    be held-out data (not the model's training data, where its scores are overconfident).
    """

    def __init__(self, model, calibrator: IsotonicRegression | None = None):
        self.model = model
        self.calibrator = calibrator

    def fit(self, X_holdout, y_holdout) -> "IsotonicCalibrated":
        raw = self.model.predict_proba(X_holdout)[:, 1]
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw, y_holdout)
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    def predict_raw(self, X) -> np.ndarray:
        """Uncalibrated P(default), for reporting the effect of calibration."""
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.calibrator.predict(self.predict_raw(X))
        return np.column_stack([1 - p, p])

    def __sklearn_is_fitted__(self) -> bool:
        return self.calibrator is not None
