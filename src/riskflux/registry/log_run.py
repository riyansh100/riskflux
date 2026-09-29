"""Log the current DVC-built model to MLflow and register it as @challenger.

DVC builds the model (reproducible, offline); this step records it (params, metrics,
lineage) and hands it to the registry. It refuses to log a model that may not match
the code: `dvc status` must be clean, and git must be clean unless --allow-dirty
(dirty versions are tagged and can never be promoted).

Run: uv run --env-file .env python -m riskflux.registry.log_run
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from riskflux import columns as C
from riskflux.config import (
    ABLATION_METRICS,
    CHALLENGER,
    EXPERIMENT_NAME,
    MODEL_PATH,
    PARAMS_PATH,
    POLICY_PATH,
    REGISTERED_MODEL,
    TEST_METRICS,
    TRAIN_METRICS,
)
from riskflux.models.io import load_split
from riskflux.registry.lineage import data_hashes, dvc_is_up_to_date, git_commit, git_is_dirty
from riskflux.registry.pyfunc import log_pyfunc

LOGGED_PARAM_SECTIONS = ("population", "split", "model", "cost")


def flatten(tree: dict, prefix: str = "") -> dict:
    """{'a': {'b': 1}} -> {'a.b': 1}"""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


def collect_metrics() -> dict[str, float]:
    reports = {"val": TRAIN_METRICS, "test": TEST_METRICS, "ablation": ABLATION_METRICS}
    metrics = {}
    for prefix, path in reports.items():
        for name, value in flatten(json.loads(path.read_text()), f"{prefix}.").items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[name] = float(value)
    return metrics


def register_challenger(
    model_path: Path,
    policy_path: Path,
    input_example: pd.DataFrame,
    params: dict,
    metrics: dict,
    tags: dict,
    report_dir: Path | None = None,
) -> str:
    """Log one run, register its model as a new version, point @challenger at it."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"train-{tags.get('git_commit', 'local')[:7]}"):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(policy_path))
        if report_dir is not None:
            mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        info = log_pyfunc(model_path, policy_path, input_example)

    version = mlflow.register_model(info.model_uri, REGISTERED_MODEL, tags=tags)
    mlflow.MlflowClient().set_registered_model_alias(REGISTERED_MODEL, CHALLENGER, version.version)
    return version.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="log despite uncommitted changes"
    )
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set: copy .env.example to .env, then use "
            "`uv run --env-file .env ...`"
        )
    if not dvc_is_up_to_date():
        raise SystemExit(
            "`dvc status` is not clean: the model may not match the code. "
            "Run `uv run dvc repro` first."
        )
    dirty = git_is_dirty()
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "Uncommitted changes: commit first (or pass --allow-dirty; "
            "the version will be tagged and can't be promoted)."
        )

    policy = json.loads(POLICY_PATH.read_text())
    tags = {
        "git_commit": git_commit(),
        "git_dirty": str(dirty).lower(),
        "threshold": str(policy["threshold"]),
        **data_hashes(),
    }
    raw_params = yaml.safe_load(PARAMS_PATH.read_text())
    params = flatten({k: raw_params[k] for k in LOGGED_PARAM_SECTIONS})
    example = load_split("test").head(5)[[C.ISSUE_DATE, *C.FEATURES]]

    version = register_challenger(
        MODEL_PATH, POLICY_PATH, example, params, collect_metrics(), tags, Path("reports")
    )
    commit = tags["git_commit"][:7]
    print(f"registered {REGISTERED_MODEL} v{version} as @{CHALLENGER} (commit {commit})")


if __name__ == "__main__":
    main()
