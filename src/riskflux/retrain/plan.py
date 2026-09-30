"""Retraining plan: move the simulated clock and derive the new training windows.

    python -m riskflux.retrain.plan --as-of 2021-03          # dry run: what would change?
    python -m riskflux.retrain.plan --as-of 2021-03 --apply  # rewrite params.yaml

Exit code 0 = windows changed (new outcomes have matured -> retrain);
exit code 3 = nothing changed (no newly labeled loans -> retraining can't help yet).
The workflow branches on this, so a drift alert with no new labels ends in a clear
"escalate" message instead of a pointless identical retrain.

params.yaml is rewritten with ruamel.yaml so comments survive: the PR diff shows
exactly which windows moved.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as Quoted

from riskflux.config import PARAMS_PATH, load_params, windows_for

NO_CHANGE = 3


def _flow(start: pd.Timestamp, end: pd.Timestamp) -> CommentedMap:
    window = CommentedMap(start=Quoted(f"{start:%Y-%m}"), end=Quoted(f"{end:%Y-%m}"))
    window.fa.set_flow_style()  # keep the one-line {start: ..., end: ...} style
    return window


def plan(as_of: str, params_path: Path = PARAMS_PATH, apply: bool = False) -> list[str]:
    """Returns human-readable changes; rewrites params.yaml if apply=True."""
    current = load_params(params_path)
    # No outcomes exist after the data snapshot, so the clock stops there.
    new_as_of = min(pd.Timestamp(as_of), pd.Timestamp(current.retraining.data_snapshot))
    if new_as_of < current.retraining.as_of:
        raise ValueError(
            f"clock can't go backwards ({new_as_of:%Y-%m} < {current.retraining.as_of:%Y-%m})"
        )
    population, splits = windows_for(new_as_of, current.retraining)

    changes = []
    if population != current.population.window:
        changes.append(
            f"population end {current.population.window.end:%Y-%m} -> {population.end:%Y-%m}"
        )
    for name, window in splits.items():
        old = current.splits[name]
        if window != old:
            before, after = (
                f"{old.start:%Y-%m}..{old.end:%Y-%m}",
                f"{window.start:%Y-%m}..{window.end:%Y-%m}",
            )
            changes.append(f"{name}: {before} -> {after}")

    if apply and new_as_of != current.retraining.as_of:
        yaml = YAML()
        yaml.preserve_quotes = True
        doc = yaml.load(params_path)
        doc["clock"]["as_of"] = Quoted(f"{new_as_of:%Y-%m}")
        doc["population"]["issue_start"] = Quoted(f"{population.start:%Y-%m}")
        doc["population"]["issue_end"] = Quoted(f"{population.end:%Y-%m}")
        for name, window in splits.items():
            doc["split"][name] = _flow(window.start, window.end)
        with params_path.open("w") as f:
            yaml.dump(doc, f)
        load_params(params_path)  # the rewritten file must pass every config check
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True, help="simulated today, YYYY-MM")
    parser.add_argument("--apply", action="store_true", help="rewrite params.yaml")
    args = parser.parse_args()

    changes = plan(args.as_of, apply=args.apply)
    if not changes:
        print(f"as_of {args.as_of}: no newly matured loans since the last training")
        sys.exit(NO_CHANGE)
    print(f"as_of {args.as_of}: new outcomes available -> retrain")
    for change in changes:
        print(f"  {change}")


if __name__ == "__main__":
    main()
