"""sklearn feature pipeline: the one object that both training and serving call.

    typed frame -> FeatureEngineer (stateless) -> CategoricalVocab (fitted on train) -> model input

Step 6 appends the model as the last step, so one fitted Pipeline (preprocessing +
model) is saved and served. Serving can't apply a different transformation, because
there is no separate preprocessing code to drift.

Deliberately NOT here: imputation and scaling. LightGBM handles NaN natively and
splits on thresholds, so neither changes its predictions. Each fitted step is state
that could go stale or differ between environments; this pipeline keeps only one.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from riskflux.features.engineer import MODEL_CATEGORICAL, engineer


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """sklearn wrapper around the stateless engineer() function."""

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True  # nothing to learn


class CategoricalVocab(TransformerMixin, BaseEstimator):
    """Fixes each categorical column to the categories seen in training.

    LightGBM needs identical category codes at train and serve time; pandas assigns
    codes from the category list, so the list itself is the learned state. A category
    never seen in training becomes NaN (treated as missing), not a crash and not a
    silently re-numbered code.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalVocab":
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in self.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, categories in self.categories_.items():
            # Mask unseen values explicitly: pandas 4 will raise instead of coercing to NaN.
            known = X[c].where(X[c].isin(categories))
            X[c] = pd.Categorical(known, categories=categories)
        return X


def build_feature_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("engineer", FeatureEngineer()),
            ("categories", CategoricalVocab(columns=MODEL_CATEGORICAL)),
        ]
    )
