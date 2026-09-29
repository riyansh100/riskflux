"""Evaluate stage: the single look at the test split.

Uses the threshold and cost model frozen in policy.json (chosen on val). Nothing here
feeds back into training, so the test numbers are an honest estimate of performance
on loans issued after everything the model or its policy saw.

Baseline: every loan in this dataset was approved by Lending Club, so "approve all"
is literally the status quo. Savings are measured against it.

Run: uv run python -m riskflux.models.evaluate
"""

from riskflux import columns as C
from riskflux.config import CALIBRATION_CURVE, TEST_METRICS, load_params
from riskflux.economics.cost import (
    CostModel,
    best_threshold,
    cost_curve,
    decision_cost,
    expected_value_reject,
)
from riskflux.models.io import aux_for, load_model, load_split, write_json
from riskflux.models.metrics import calibration_table, probability_metrics


def main() -> None:
    params = load_params()
    pipeline, policy = load_model()
    test = load_split("test")
    y = test[C.TARGET].to_numpy()

    cost_model = CostModel(**policy["cost_model"])
    test_aux = aux_for(test)
    loss, profit = cost_model.loss_if_default(test_aux), cost_model.profit_if_good(test_aux)

    pd_test = pipeline.predict_proba(test)[:, 1]
    pd_raw = pipeline[-1].predict_raw(pipeline[:-1].transform(test))

    def summarize(reject) -> dict:
        cost = decision_cost(y, reject, loss, profit)
        return {
            "cost": round(cost),
            "savings": round(approve_all - cost),
            "savings_pct": round(100 * (1 - cost / approve_all), 2),
            "reject_rate": round(float(reject.mean()), 4),
            "defaults_caught": round(float(reject[y == 1].mean()), 4),
            "good_loans_rejected": round(float(reject[y == 0].mean()), 4),
        }

    approve_all = decision_cost(y, pd_test > 1, loss, profit)
    # Hindsight only: the best threshold had we been allowed to tune on test.
    # Shows how much the val-chosen threshold leaves on the table; never used for decisions.
    oracle = best_threshold(cost_curve(y, pd_test, loss, profit, params.cost.threshold_grid))

    metrics = {
        "loans": len(test),
        "default_rate": round(float(y.mean()), 5),
        "calibrated": probability_metrics(y, pd_test),
        "raw": probability_metrics(y, pd_raw),
        "cost_approve_all": round(approve_all),
        "policy_threshold": {
            "threshold": policy["threshold"],
            **summarize(pd_test >= policy["threshold"]),
        },
        "policy_expected_value": summarize(expected_value_reject(pd_test, loss, profit)),
        "hindsight_best_threshold": {"threshold": oracle, **summarize(pd_test >= oracle)},
    }
    write_json(TEST_METRICS, metrics)
    calibration_table(y, pd_test).round(5).to_csv(CALIBRATION_CURVE, index=False)

    chosen = metrics["policy_threshold"]
    print(
        f"test AUC={metrics['calibrated']['roc_auc']}  ECE={metrics['calibrated']['ece']}  "
        f"threshold={policy['threshold']}  "
        f"savings=${chosen['savings']:,} ({chosen['savings_pct']}%)"
    )


if __name__ == "__main__":
    main()
