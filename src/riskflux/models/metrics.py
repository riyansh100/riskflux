"""Model quality metrics: ranking (AUC), probability quality (Brier, ECE), calibration curve."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_table(y: np.ndarray, pd_hat: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Loans sorted into equal-size PD bins: predicted vs observed default rate per bin."""
    frame = pd.DataFrame({"y": y, "pd": pd_hat})
    frame["bin"] = pd.qcut(frame["pd"].rank(method="first"), bins, labels=False)
    return (
        frame.groupby("bin")
        .agg(mean_predicted=("pd", "mean"), observed_rate=("y", "mean"), loans=("y", "size"))
        .reset_index()
    )


def expected_calibration_error(y: np.ndarray, pd_hat: np.ndarray, bins: int = 10) -> float:
    """Loan-weighted average gap between predicted and observed default rate across bins."""
    table = calibration_table(y, pd_hat, bins)
    gaps = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float((gaps * table["loans"]).sum() / table["loans"].sum())


def probability_metrics(y: np.ndarray, pd_hat: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": round(float(roc_auc_score(y, pd_hat)), 5),
        "pr_auc": round(float(average_precision_score(y, pd_hat)), 5),
        "brier": round(float(brier_score_loss(y, pd_hat)), 5),
        "log_loss": round(float(log_loss(y, np.clip(pd_hat, 1e-6, 1 - 1e-6))), 5),
        "ece": round(expected_calibration_error(y, pd_hat), 5),
    }
