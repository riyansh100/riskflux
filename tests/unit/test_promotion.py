import pytest

from riskflux.config import PromotionParams
from riskflux.registry.log_run import flatten
from riskflux.registry.promote import Evaluation, decide

RULES = PromotionParams(min_cost_improvement=0.001, max_auc_drop=0.005, max_ece=0.02)
CHAMPION = Evaluation(version="1", cost=1_000_000.0, roc_auc=0.690, ece=0.005)


def challenger(**overrides) -> Evaluation:
    return Evaluation(
        **{"version": "2", "cost": 990_000.0, "roc_auc": 0.690, "ece": 0.005} | overrides
    )


def test_first_model_is_promoted_if_calibrated():
    assert decide(challenger(), None, RULES)[0]


def test_uncalibrated_model_is_never_promoted_even_as_first():
    promoted, reason = decide(challenger(ece=0.05), None, RULES)
    assert not promoted and "ECE" in reason


def test_cheaper_model_is_promoted():
    promoted, reason = decide(challenger(cost=990_000.0), CHAMPION, RULES)  # 1% cheaper
    assert promoted and "1.000% lower" in reason


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"cost": 1_000_000.0}, "improvement"),  # equal cost: no churn for nothing
        ({"cost": 999_500.0}, "improvement"),  # 0.05% cheaper: below the 0.1% bar
        ({"cost": 900_000.0, "roc_auc": 0.680}, "ROC-AUC"),  # cheaper but ranks worse
    ],
)
def test_guardrails_block_promotion(overrides, why):
    promoted, reason = decide(challenger(**overrides), CHAMPION, RULES)
    assert not promoted and why in reason


def test_flatten_nested_params():
    assert flatten({"model": {"lightgbm": {"num_leaves": 31}, "seed": 42}}) == {
        "model.lightgbm.num_leaves": 31,
        "model.seed": 42,
    }
