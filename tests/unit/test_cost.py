import numpy as np
import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.economics.cost import (
    CostModel,
    best_threshold,
    contract_interest,
    cost_curve,
    decision_cost,
    estimate_cost_model,
    expected_value_reject,
)

# Two defaults and two repaid loans with hand-computable outcomes.
LOANS = pd.DataFrame({C.ID: [1, 2, 3, 4], C.TARGET: [1, 1, 0, 0]})
AUX = pd.DataFrame(
    {
        C.ID: [1, 2, 3, 4],
        "funded_amnt": [10_000.0, 30_000.0, 10_000.0, 10_000.0],
        "total_rec_prncp": [4_000.0, 12_000.0, 10_000.0, 10_000.0],
        "recoveries": [1_000.0, 0.0, 0.0, 0.0],
        "collection_recovery_fee": [0.0, 0.0, 0.0, 0.0],
        "installment": [300.0, 900.0, 300.0, 300.0],  # contract interest = 800 per $10k
        "total_rec_int": [0.0, 0.0, 800.0, 400.0],
    }
)


def test_lgd_and_haircut_are_dollar_weighted():
    model = estimate_cost_model(LOANS, AUX, term_months=36)
    # lost: 10k-5k=5k and 30k-12k=18k -> 23k / 40k funded (mean of ratios would be 0.55)
    assert model.lgd == pytest.approx(23 / 40)
    # collected 1,200 of 1,600 contractual interest
    assert model.haircut == pytest.approx(0.75)


def test_contract_interest():
    assert contract_interest(AUX, 36).tolist() == [800.0, 2_400.0, 800.0, 800.0]


def test_decision_cost_counts_only_the_two_kinds_of_mistake():
    y = np.array([1, 1, 0, 0])
    loss, profit = np.array([10.0, 20.0, 99.0, 99.0]), np.array([99.0, 99.0, 3.0, 4.0])
    # approve everything: pay for both defaults
    assert decision_cost(y, np.zeros(4, dtype=bool), loss, profit) == 30.0
    # reject everything: forgo both good loans' profit
    assert decision_cost(y, np.ones(4, dtype=bool), loss, profit) == 7.0
    # perfect decisions cost nothing
    assert decision_cost(y, y == 1, loss, profit) == 0.0


def test_best_threshold_minimizes_cost():
    y = np.array([1, 0, 0, 1])
    pd_hat = np.array([0.9, 0.1, 0.2, 0.6])
    loss, profit = np.full(4, 100.0), np.full(4, 10.0)
    curve = cost_curve(y, pd_hat, loss, profit, np.array([0.05, 0.5, 0.95]))
    # 0.05 rejects all (forgo 2 x $10), 0.5 rejects exactly the two defaults ($0),
    # 0.95 rejects none (both $100 defaults approved)
    assert curve["cost"].tolist() == [20.0, 0.0, 200.0]
    assert best_threshold(curve) == 0.5


def test_expected_value_rule_uses_each_loans_own_break_even():
    # Same PD, different economics: big loss -> reject; small loss -> approve.
    pd_hat = np.array([0.2, 0.2])
    reject = expected_value_reject(pd_hat, np.array([5_000.0, 500.0]), np.array([1_000.0, 1_000.0]))
    assert reject.tolist() == [True, False]


def test_cost_model_round_trips_through_policy_json():
    model = CostModel(lgd=0.5, haircut=0.8, term_months=36)
    assert CostModel(**model.to_dict()) == model
