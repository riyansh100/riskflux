"""Retraining backtest: what would a yearly retrain have been worth?

For each simulated date in params (backtest.as_of) the exact production training
procedure is run on the loans whose outcomes were known then. Every model is then
scored on every LATER matured window, so the table shows:
  - decay: how a frozen model does on newer and newer loans
  - value of retraining: frozen old model vs the model retrained at that window's date

Each window is priced with the economics estimated at that window's date (the cost
model of the model retrained then), so all models in a column face the same prices.

Run: uv run python -m riskflux.retrain.backtest
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from riskflux import columns as C
from riskflux.config import AUX_PARQUET, BACKTEST_METRICS, CLEAN_PARQUET, load_params, windows_for
from riskflux.economics.cost import CostModel, decision_cost
from riskflux.models.io import aux_for, write_json
from riskflux.models.metrics import expected_calibration_error
from riskflux.models.train import fit_policy_model


def main() -> None:
    params = load_params()
    loans, aux = pd.read_parquet(CLEAN_PARQUET), pd.read_parquet(AUX_PARQUET)
    issued = loans[C.ISSUE_DATE]

    trained = {}
    for as_of in params.retraining.backtest_as_of:
        _, splits = windows_for(as_of, params.retraining)
        train, val = loans[splits["train"].contains(issued)], loans[splits["val"].contains(issued)]
        pipeline, policy, _ = fit_policy_model(
            train, val, aux_for(train, aux), aux_for(val, aux), params
        )
        trained[as_of] = {"pipeline": pipeline, "policy": policy, "test": splits["test"]}
        print(f"trained as of {as_of}: {len(train):,} loans, threshold {policy['threshold']}")

    rows = []
    for eval_as_of, judge in trained.items():
        window = judge["test"]
        test = loans[window.contains(issued)]
        test_aux, y = aux_for(test, aux), test[C.TARGET].to_numpy()
        prices = CostModel(**judge["policy"]["cost_model"])
        loss, profit = prices.loss_if_default(test_aux), prices.profit_if_good(test_aux)
        approve_all = decision_cost(y, y < 0, loss, profit)
        for model_as_of, model in trained.items():
            if model["test"].start > window.start:
                continue  # a model can't be evaluated on loans from before it was trained
            pd_hat = model["pipeline"].predict_proba(test)[:, 1]
            reject = pd_hat >= model["policy"]["threshold"]
            cost = decision_cost(y, reject, loss, profit)
            rows.append(
                {
                    "model_as_of": model_as_of,
                    "evaluated_on": f"{window.start:%Y-%m}..{window.end:%Y-%m}",
                    "eval_as_of": eval_as_of,
                    "roc_auc": round(float(roc_auc_score(y, pd_hat)), 4),
                    "ece": round(expected_calibration_error(y, pd_hat), 4),
                    "savings_pct": round(100 * (1 - cost / approve_all), 2),
                    "reject_rate": round(float(reject.mean()), 4),
                }
            )

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    write_json(BACKTEST_METRICS, {"rows": rows})


if __name__ == "__main__":
    main()
