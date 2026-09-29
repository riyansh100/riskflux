"""Where did this model come from? Git commit + DVC data hashes + pipeline freshness."""

import json
import subprocess
from pathlib import Path

import yaml

DVC_LOCK = Path("dvc.lock")
LINEAGE_OUTPUTS = {  # stage outputs whose content hashes identify the training data
    "data.train_md5": "data/processed/train.parquet",
    "data.val_md5": "data/processed/val.parquet",
    "data.test_md5": "data/processed/test.parquet",
    "data.aux_md5": "data/interim/loans_aux.parquet",
    "model.joblib_md5": "models/model.joblib",
}


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def git_commit() -> str:
    return _run("git", "rev-parse", "HEAD")


def git_is_dirty() -> bool:
    """Uncommitted changes mean the commit hash doesn't fully describe the code."""
    return bool(_run("git", "status", "--porcelain"))


def dvc_is_up_to_date() -> bool:
    """False if any stage's code, params or data changed since its outputs were built,
    i.e. models/model.joblib may not correspond to the current commit."""
    return json.loads(_run("dvc", "status", "--json") or "{}") == {}


def data_hashes(lock_path: Path = DVC_LOCK) -> dict[str, str]:
    lock = yaml.safe_load(lock_path.read_text())
    outputs = {
        out["path"]: out["md5"]
        for stage in lock["stages"].values()
        for out in stage.get("outs", [])
        if "md5" in out
    }
    return {tag: outputs[path] for tag, path in LINEAGE_OUTPUTS.items()}
