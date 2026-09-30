"""Champion/challenger promotion.

Scores @challenger and @champion on the SAME labeled loans with the SAME economics,
and moves @champion to the challenger only if it is cheaper in dollars and passes the
guardrails in params.yaml (promotion). Serving follows @champion, so this is the
single gate between "trained" and "in production". Rollback = point @champion back.

Run: uv run --env-file .env python -m riskflux.registry.promote
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from riskflux import columns as C
from riskflux.config import CHALLENGER, CHAMPION, REGISTERED_MODEL, PromotionParams, load_params
from riskflux.economics.cost import CostModel, decision_cost
from riskflux.models.io import aux_for, load_split
from riskflux.models.metrics import expected_calibration_error
from riskflux.registry.lineage import data_hashes


@dataclass(frozen=True)
class Evaluation:
    version: str
    cost: float
    roc_auc: float
    ece: float


def decide(
    challenger: Evaluation, champion: Evaluation | None, rules: PromotionParams
) -> tuple[bool, str]:
    if challenger.ece > rules.max_ece:
        return False, f"challenger ECE {challenger.ece:.4f} exceeds {rules.max_ece}"
    if champion is None:
        return True, "no champion yet (bootstrap)"
    if challenger.roc_auc < champion.roc_auc - rules.max_auc_drop:
        return False, (
            f"ROC-AUC dropped {champion.roc_auc - challenger.roc_auc:.4f} "
            f"(max {rules.max_auc_drop})"
        )
    improvement = 1 - challenger.cost / champion.cost
    if improvement < rules.min_cost_improvement:
        return False, f"cost improvement {improvement:.3%} below {rules.min_cost_improvement:.3%}"
    return True, f"cost {improvement:.3%} lower than champion v{champion.version}"


def evaluate_version(
    version: str, loans: pd.DataFrame, aux: pd.DataFrame, cost_model: CostModel
) -> Evaluation:
    """Score one registered version with its own threshold, priced by `cost_model`."""
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    scored = model.predict(loans[[C.ISSUE_DATE, *C.FEATURES]])
    y = loans[C.TARGET].to_numpy()
    cost = decision_cost(
        y,
        scored["reject"].to_numpy(dtype=bool),
        cost_model.loss_if_default(aux),
        cost_model.profit_if_good(aux),
    )
    pd_hat = scored["pd"].to_numpy(dtype=np.float64)
    return Evaluation(
        version=version,
        cost=cost,
        roc_auc=float(roc_auc_score(y, pd_hat)),
        ece=expected_calibration_error(y, pd_hat),
    )


def promote(
    loans: pd.DataFrame,
    aux: pd.DataFrame,
    rules: PromotionParams,
    dry_run: bool = False,
    expected_model_md5: str | None = None,
) -> tuple[bool, str, list[Evaluation]]:
    """Returns (promoted, reason, evaluations). dry_run: decide and report, change nothing.
    expected_model_md5: the challenger must be exactly the model in this commit's dvc.lock
    (content check, so it holds after squash merges or history rewrites)."""
    client = mlflow.MlflowClient()
    challenger = client.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER)
    if challenger.tags.get("git_dirty") == "true":
        return False, f"v{challenger.version} was logged from uncommitted code", []
    if expected_model_md5 and challenger.tags.get("model.joblib_md5") != expected_model_md5:
        return False, f"v{challenger.version} is not the model in this commit's dvc.lock", []
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
    except MlflowException:
        champion = None
    if champion is not None and champion.version == challenger.version:
        return False, f"v{challenger.version} is already the champion", []

    # Judge both with the challenger's economics (the most recent cost estimates).
    policy_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{challenger.version}")
    cost_model = CostModel(**policy_model.unwrap_python_model().policy["cost_model"])

    challenger_eval = evaluate_version(challenger.version, loans, aux, cost_model)
    champion_eval = evaluate_version(champion.version, loans, aux, cost_model) if champion else None
    promoted, reason = decide(challenger_eval, champion_eval, rules)
    evaluations = [e for e in (challenger_eval, champion_eval) if e is not None]
    if dry_run:
        return promoted, reason, evaluations

    client.set_model_version_tag(REGISTERED_MODEL, challenger.version, "promotion_decision", reason)
    if promoted:
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, challenger.version)
        client.set_model_version_tag(
            REGISTERED_MODEL, challenger.version, "promoted_at", datetime.now(UTC).isoformat()
        )
    return promoted, reason, evaluations


def markdown_report(
    promoted: bool, reason: str, evaluations: list[Evaluation], dry_run: bool
) -> str:
    verdict = ("would be PROMOTED" if dry_run else "PROMOTED") if promoted else "NOT promoted"
    lines = [f"**Promotion decision: {verdict}** — {reason}", ""]
    if evaluations:
        lines += ["| role | version | cost ($) | ROC-AUC | ECE |", "|---|---|---|---|---|"]
        for role, e in zip(("challenger", "champion"), evaluations, strict=False):
            lines.append(
                f"| {role} | v{e.version} | {e.cost:,.0f} | {e.roc_auc:.4f} | {e.ece:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="decide and report only")
    parser.add_argument(
        "--require-lock-match", action="store_true", help="challenger must match dvc.lock"
    )
    parser.add_argument("--report", type=Path, help="write a markdown summary here")
    args = parser.parse_args()

    params = load_params()
    # Promotion data: the newest matured window under the current clock (= test split).
    # The champion was trained at an earlier clock and has never seen these loans.
    loans = load_split("test")
    expected = data_hashes()["model.joblib_md5"] if args.require_lock_match else None
    promoted, reason, evaluations = promote(
        loans, aux_for(loans), params.promotion, args.dry_run, expected
    )
    report = markdown_report(promoted, reason, evaluations, args.dry_run)
    print(report)
    if args.report:
        args.report.write_text(report)


if __name__ == "__main__":
    main()
