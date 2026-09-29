"""The business cost function (D5).

Every approved-or-rejected decision is priced in dollars, relative to a perfect lender:

    approve a loan that defaults   -> lose  LGD x funded_amnt
    reject a loan that would repay -> forgo haircut x contractual interest
    approve good / reject bad      -> 0

LGD and the haircut are estimated from TRAIN loans only (their outcomes are known),
then frozen into the policy. Estimating them on val/test would leak outcomes.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from riskflux import columns as C


@dataclass(frozen=True)
class CostModel:
    lgd: float  # share of funded principal lost when a loan defaults
    haircut: float  # share of contractual interest actually collected on good loans
    term_months: int

    def loss_if_default(self, aux: pd.DataFrame) -> np.ndarray:
        return (self.lgd * aux["funded_amnt"]).to_numpy()

    def profit_if_good(self, aux: pd.DataFrame) -> np.ndarray:
        return (self.haircut * contract_interest(aux, self.term_months)).to_numpy()

    def to_dict(self) -> dict:
        return asdict(self)


def contract_interest(aux: pd.DataFrame, term_months: int) -> pd.Series:
    """Total interest if the borrower pays every installment on schedule."""
    return aux["installment"] * term_months - aux["funded_amnt"]


def estimate_cost_model(loans: pd.DataFrame, aux: pd.DataFrame, term_months: int) -> CostModel:
    """Dollar-weighted estimates (ratio of sums, not mean of ratios): the cost function
    sums dollars, so a $35k loan should weigh 17x more than a $2k one."""
    joined = loans[[C.ID, C.TARGET]].merge(aux, on=C.ID, how="left", validate="1:1")
    bad, good = joined[joined[C.TARGET] == 1], joined[joined[C.TARGET] == 0]

    recovered = bad["total_rec_prncp"] + bad["recoveries"] - bad["collection_recovery_fee"]
    lgd = 1 - recovered.sum() / bad["funded_amnt"].sum()
    haircut = good["total_rec_int"].sum() / contract_interest(good, term_months).sum()
    return CostModel(
        lgd=round(float(lgd), 6), haircut=round(float(haircut), 6), term_months=term_months
    )


def decision_cost(
    y: np.ndarray, reject: np.ndarray, loss_if_default: np.ndarray, profit_if_good: np.ndarray
) -> float:
    approved_bad = ~reject & (y == 1)
    rejected_good = reject & (y == 0)
    return float(loss_if_default[approved_bad].sum() + profit_if_good[rejected_good].sum())


def cost_curve(
    y: np.ndarray,
    pd_hat: np.ndarray,
    loss_if_default: np.ndarray,
    profit_if_good: np.ndarray,
    grid: np.ndarray,
) -> pd.DataFrame:
    """Total cost of the rule 'reject if PD >= t' for every threshold t in grid."""
    rows = [
        (t, decision_cost(y, pd_hat >= t, loss_if_default, profit_if_good), (pd_hat >= t).mean())
        for t in grid
    ]
    return pd.DataFrame(rows, columns=["threshold", "cost", "reject_rate"])


def best_threshold(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def expected_value_reject(
    pd_hat: np.ndarray, loss_if_default: np.ndarray, profit_if_good: np.ndarray
) -> np.ndarray:
    """Per-loan rule: reject when expected loss exceeds expected profit,
    i.e. PD x L > (1 - PD) x G. Each loan gets its own break-even PD = G / (L + G)."""
    return pd_hat * loss_if_default > (1 - pd_hat) * profit_if_good
