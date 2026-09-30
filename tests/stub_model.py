"""Build a tiny stand-in model folder (same format as fetch_model output) from synthetic
loans, so CI can build and smoke-test the Docker image without registry credentials.

Run: uv run python -m tests.stub_model build/model
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.io import save_model
from tests.conftest import make_clean_frame


def build(out_dir: Path) -> None:
    train, holdout = make_clean_frame(400, seed=1), make_clean_frame(300, seed=2)
    features = build_feature_pipeline().fit(train)
    lgbm = lgb.LGBMClassifier(n_estimators=20, min_child_samples=5, verbose=-1, random_state=0)
    lgbm.fit(features.transform(train), train[C.TARGET])
    calibrated = IsotonicCalibrated(lgbm).fit(features.transform(holdout), holdout[C.TARGET])
    policy = {"threshold": 0.24, "cost_model": {"lgd": 0.5, "haircut": 0.8, "term_months": 36}}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        Pipeline([*features.steps, ("model", calibrated)]),
        policy,
        out_dir / "model.joblib",
        out_dir / "policy.json",
    )
    (out_dir / "metadata.json").write_text(json.dumps({"version": "stub"}) + "\n")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "build/model"))
    print("stub model written")
