"""Download a registered model (default: @champion) into a folder for the Docker build.

Runs OUTSIDE `docker build`, so registry credentials never enter the image. The
download is verified: its md5 must equal the `model.joblib_md5` lineage tag recorded
at registration (i.e. the exact file DVC built). Writes metadata.json so the running
service can report which version it serves.

Run: uv run --env-file .env python -m riskflux.serving.fetch_model [--alias champion]
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlflow

from riskflux.config import CHAMPION, REGISTERED_MODEL

DEFAULT_OUT = Path("build/model")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(alias: str, out_dir: Path) -> dict:
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{REGISTERED_MODEL}@{alias}"))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name in ("model.joblib", "policy.json"):
        shutil.copy2(local / "artifacts" / name, out_dir / name)

    expected = version.tags.get("model.joblib_md5")
    actual = md5(out_dir / "model.joblib")
    if expected != actual:
        raise SystemExit(f"model.joblib md5 {actual} != registry lineage tag {expected}")

    metadata = {
        "registered_model": REGISTERED_MODEL,
        "version": version.version,
        "alias": alias,
        "run_id": version.run_id,
        "git_commit": version.tags.get("git_commit"),
        "model_md5": actual,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alias", default=CHAMPION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    meta = fetch(args.alias, args.out)
    print(
        f"fetched {meta['registered_model']} v{meta['version']} (@{meta['alias']}, "
        f"commit {meta['git_commit'][:7]}, md5 verified) -> {args.out}"
    )


if __name__ == "__main__":
    main()
