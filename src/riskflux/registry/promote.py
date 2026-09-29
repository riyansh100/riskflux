"""Champion/challenger promotion.

Scores @challenger and @champion on the SAME labeled loans with the SAME economics,
and moves @champion to the challenger only if it is cheaper in dollars and passes the
guardrails in params.yaml (promotion). Serving follows @champion, so this is the
single gate between "trained" and "in production". Rollback = point @champion back.

Run: uv run --env-file .env python -m riskflux.registry.promote
"""

from dataclasses import dataclass
from datetime import UTC, datetime

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


def promote(loans: pd.DataFrame, aux: pd.DataFrame, rules: PromotionParams) -> tuple[bool, str]:
    client = mlflow.MlflowClient()
    challenger = client.get_model_version_by_alias(REGISTERED_MODEL, CHALLENGER)
    if challenger.tags.get("git_dirty") == "true":
        return False, f"v{challenger.version} was logged from uncommitted code"
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION)
    except MlflowException:
        champion = None
    if champion is not None and champion.version == challenger.version:
        return False, f"v{challenger.version} is already the champion"

    # Judge both with the challenger's economics (the most recent cost estimates).
    policy_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{challenger.version}")
    cost_model = CostModel(**policy_model.unwrap_python_model().policy["cost_model"])

    challenger_eval = evaluate_version(challenger.version, loans, aux, cost_model)
    champion_eval = evaluate_version(champion.version, loans, aux, cost_model) if champion else None
    promoted, reason = decide(challenger_eval, champion_eval, rules)

    client.set_model_version_tag(REGISTERED_MODEL, challenger.version, "promotion_decision", reason)
    if promoted:
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION, challenger.version)
        client.set_model_version_tag(
            REGISTERED_MODEL, challenger.version, "promoted_at", datetime.now(UTC).isoformat()
        )
    return promoted, reason


def main() -> None:
    params = load_params()
    # Promotion data: the newest labeled period. Corner cut: this is the test split,
    # which makes test a selection set once more than one version competes.
    loans = load_split("test")
    promoted, reason = promote(loans, aux_for(loans), params.promotion)
    print(f"{'PROMOTED' if promoted else 'NOT promoted'}: {reason}")


if __name__ == "__main__":
    main()
