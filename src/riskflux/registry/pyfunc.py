"""The registered model format: an MLflow pyfunc wrapping model.joblib + policy.json.

Input: a typed loan frame (issue_d + FEATURES), as produced by the clean stage or by
applications_to_frame at the API. Output: calibrated PD and the policy decision.

The serving image loads model.joblib + policy.json directly (no MLflow at runtime, D10);
the pyfunc wrapper is what the registry stores, so any version can also be loaded and
scored generically, e.g. by the promotion step.
"""

from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd

import riskflux
from riskflux.models.io import load_model

# Pinned into the logged model so it can be rebuilt in a fresh environment.
RUNTIME_PACKAGES = [
    "lightgbm", "scikit-learn", "pandas", "numpy", "pyarrow", "pandera", "pydantic",
    "pyyaml", "joblib", "mlflow-skinny",
]  # fmt: skip


class RiskFluxModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.pipeline, self.policy = load_model(
            Path(context.artifacts["model"]), Path(context.artifacts["policy"])
        )

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        pd_hat = self.pipeline.predict_proba(model_input)[:, 1]
        return pd.DataFrame({"pd": pd_hat, "reject": pd_hat >= self.policy["threshold"]})


def log_pyfunc(model_path: Path, policy_path: Path, input_example: pd.DataFrame):
    """Log inside an active run. Returns MLflow's ModelInfo."""
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=RiskFluxModel(),
        artifacts={"model": str(model_path), "policy": str(policy_path)},
        # Snapshot of the riskflux code, so the pickled pipeline's classes can be imported.
        code_paths=[str(Path(riskflux.__file__).parent)],
        input_example=input_example,
        pip_requirements=[f"{pkg}=={version(pkg)}" for pkg in RUNTIME_PACKAGES],
    )
