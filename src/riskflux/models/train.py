"""Train stage: train + val -> fitted pipeline and decision policy.

    1. Cost model (LGD, haircut) from TRAIN outcomes only.
    2. Feature pipeline fitted on train.
    3. LightGBM on train, early-stopped on val log-loss. No resampling or class weights (D6):
       they would distort the probabilities the cost function needs.
    4. Isotonic calibration fitted on val predictions.
    5. Threshold minimizing total dollar cost on val.

The test split is never read here.

Run: uv run python -m riskflux.models.train
"""

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from riskflux import columns as C
from riskflux.config import TRAIN_METRICS, load_params
from riskflux.economics.cost import (
    best_threshold,
    cost_curve,
    decision_cost,
    estimate_cost_model,
    expected_value_reject,
)
from riskflux.features.engineer import MODEL_FEATURES
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.models.calibration import IsotonicCalibrated
from riskflux.models.io import aux_for, load_split, save_model, write_json
from riskflux.models.metrics import probability_metrics


def fit_lightgbm(X_train, y_train, X_val, y_val, params) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(random_state=params.seed, **params.lightgbm)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(params.early_stopping_rounds, verbose=False)],
    )
    return model


def fit_policy_model(train, val, train_aux, val_aux, params) -> tuple[Pipeline, dict, dict]:
    """The full training procedure: cost model (train only) -> features -> LightGBM
    (early-stopped on val) -> isotonic calibration (val) -> cost-optimal threshold (val).
    Returns (pipeline, policy, metrics). Shared by the train stage and the backtest."""
    y_train, y_val = train[C.TARGET].to_numpy(), val[C.TARGET].to_numpy()
    cost_model = estimate_cost_model(train, train_aux, params.population.term_months)

    features = build_feature_pipeline().fit(train)
    X_train, X_val = features.transform(train), features.transform(val)
    lgbm = fit_lightgbm(X_train, y_train, X_val, y_val, params.model)
    calibrated = IsotonicCalibrated(lgbm).fit(X_val, y_val)
    pipeline = Pipeline([*features.steps, ("model", calibrated)])

    loss, profit = cost_model.loss_if_default(val_aux), cost_model.profit_if_good(val_aux)
    pd_val = pipeline.predict_proba(val)[:, 1]
    curve = cost_curve(y_val, pd_val, loss, profit, params.cost.threshold_grid)
    threshold = best_threshold(curve)

    policy = {
        "threshold": threshold,
        "cost_model": cost_model.to_dict(),
        "features": MODEL_FEATURES,
        "best_iteration": int(lgbm.best_iteration_),
    }

    approve_all = decision_cost(y_val, pd_val > 1, loss, profit)  # reject nobody
    chosen = curve.loc[curve["threshold"] == threshold].iloc[0]
    ev_reject = expected_value_reject(pd_val, loss, profit)
    ev_cost = decision_cost(y_val, ev_reject, loss, profit)
    metrics = {
        "best_iteration": int(lgbm.best_iteration_),
        "lgd": cost_model.lgd,
        "haircut": cost_model.haircut,
        "threshold": threshold,
        "val_raw": probability_metrics(y_val, calibrated.predict_raw(X_val)),
        "val_calibrated": probability_metrics(y_val, pd_val),
        "val_cost": {
            "approve_all": round(approve_all),
            "model": round(float(chosen["cost"])),
            "savings_pct": round(100 * (1 - chosen["cost"] / approve_all), 2),
            "reject_rate": round(float(chosen["reject_rate"]), 4),
        },
        # Alternative policy (D5c), compared on val so any switch is decided without test data.
        "val_cost_expected_value_rule": {
            "model": round(ev_cost),
            "savings_pct": round(100 * (1 - ev_cost / approve_all), 2),
            "reject_rate": round(float(ev_reject.mean()), 4),
        },
    }
    return pipeline, policy, metrics


def main() -> None:
    params = load_params()
    train, val = load_split("train"), load_split("val")
    pipeline, policy, metrics = fit_policy_model(train, val, aux_for(train), aux_for(val), params)
    save_model(pipeline, policy)
    write_json(TRAIN_METRICS, metrics)
    print(
        f"trees={metrics['best_iteration']}  val AUC={metrics['val_calibrated']['roc_auc']}  "
        f"threshold={policy['threshold']}  val savings={metrics['val_cost']['savings_pct']}%"
    )


if __name__ == "__main__":
    main()
