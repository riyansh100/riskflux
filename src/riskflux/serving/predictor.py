"""Online predictor: validated applications -> PD, decision, reason codes, policy flags.

Uses the exact fitted pipeline from training (features -> LightGBM -> isotonic), so the
score equals the batch score for the same loan (tested through HTTP in test_api.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from riskflux.economics.cost import CostModel
from riskflux.models.io import load_model
from riskflux.schemas import applications_to_frame
from riskflux.serving.reasons import reason_label

N_REASONS = 3

# Lending Club's approval box, observed in the training data (EDA + Step 4): every
# training loan satisfies these. Outside them the model is extrapolating, so the
# application is referred to a human instead of auto-decided.
POLICY_BOX = {
    "fico_below_660": lambda frame, X: frame["fico_range_low"] < 660,
    "credit_history_under_36_months": lambda frame, X: X["credit_history_months"] < 36,
    "loan_to_income_above_0.5": lambda frame, X: X["loan_to_income"] > 0.5,
}


@dataclass(frozen=True)
class ModelBundle:
    pipeline: object
    policy: dict
    metadata: dict  # registry version, git commit (absent for a locally built model)

    @classmethod
    def load(cls, model_dir: Path) -> "ModelBundle":
        pipeline, policy = load_model(model_dir / "model.joblib", model_dir / "policy.json")
        meta_path = model_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(pipeline=pipeline, policy=policy, metadata=metadata)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "local"))


class Predictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.features = bundle.pipeline[:-1]  # FeatureEngineer -> CategoricalVocab
        self.calibrated = bundle.pipeline[-1]  # IsotonicCalibrated(LightGBM)
        self.threshold = float(bundle.policy["threshold"])
        self.cost = CostModel(**bundle.policy["cost_model"])

    def predict(self, applications: list) -> list[dict]:
        frame = applications_to_frame(applications)
        X = self.features.transform(frame)
        pd_hat = self.calibrated.predict_proba(X)[:, 1]
        # TreeSHAP contributions in log-odds space; last column is the bias term.
        contrib = self.calibrated.model.predict(X, pred_contrib=True)[:, :-1]
        flags = {name: rule(frame, X).to_numpy() for name, rule in POLICY_BOX.items()}

        results = []
        for i in range(len(frame)):
            row_flags = [name for name, hits in flags.items() if hits[i]]
            results.append(
                {
                    "pd": float(pd_hat[i]),
                    "decision": self._decide(pd_hat[i], row_flags),
                    "threshold": self.threshold,
                    # Requested amount stands in for funded amount (unknown before funding).
                    "expected_loss": round(
                        float(pd_hat[i] * self.cost.lgd * frame["loan_amnt"].iloc[i]), 2
                    ),
                    "risk_factors": self._risk_factors(contrib[i], X.iloc[i]),
                    "policy_flags": row_flags,
                    "model_version": self.bundle.version,
                }
            )
        return results

    def _decide(self, pd_hat: float, flags: list[str]) -> str:
        if flags:
            return "refer"
        return "reject" if pd_hat >= self.threshold else "approve"

    @staticmethod
    def _risk_factors(contributions: np.ndarray, values: pd.Series) -> list[dict]:
        """Top features pushing this applicant's risk UP (positive contribution).
        A missing value can raise risk too (thin file); say so instead of implying
        the value itself was bad."""
        order = np.argsort(contributions)[::-1][:N_REASONS]
        factors = []
        for j in order:
            if contributions[j] <= 0:
                break
            name = values.index[j]
            label = reason_label(name)
            factors.append(
                {
                    "feature": name,
                    "description": f"{label} (not provided)" if pd.isna(values.iloc[j]) else label,
                    "contribution": round(float(contributions[j]), 4),
                }
            )
        return factors
