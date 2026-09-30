"""Plain-language labels for reason codes (the "why" behind a risk score).

US lenders must give specific reasons when they deny credit (ECOA / Regulation B
adverse action notices). Features without a label fall back to their column name.
"""

REASON_LABELS = {
    "fico_mid": "Credit score",
    "dti": "Debt-to-income ratio",
    "loan_to_income": "Loan amount relative to income",
    "revol_bal_to_income": "Revolving balance relative to income",
    "annual_inc": "Income",
    "loan_amnt": "Loan amount",
    "emp_length": "Length of employment",
    "home_ownership": "Home ownership status",
    "purpose": "Loan purpose",
    "credit_history_months": "Length of credit history",
    "revol_util": "Revolving credit utilization",
    "bc_util": "Bankcard utilization",
    "percent_bc_gt_75": "Share of bankcards above 75% utilization",
    "bc_open_to_buy": "Available bankcard credit",
    "inq_last_6mths": "Credit inquiries in the last 6 months",
    "mths_since_recent_inq": "Time since most recent credit inquiry",
    "acc_open_past_24mths": "Accounts opened in the last 24 months",
    "num_tl_op_past_12m": "Accounts opened in the last 12 months",
    "mo_sin_rcnt_tl": "Time since most recent account opened",
    "mo_sin_old_rev_tl_op": "Age of oldest revolving account",
    "mort_acc": "Number of mortgage accounts",
    "delinq_2yrs": "Delinquencies in the last 2 years",
    "mths_since_last_delinq": "Time since most recent delinquency",
    "pub_rec": "Derogatory public records",
    "pub_rec_bankruptcies": "Bankruptcies on record",
    "tot_hi_cred_lim": "Total credit limit",
    "total_bc_limit": "Total bankcard limit",
    "total_rev_hi_lim": "Total revolving credit limit",
    "avg_cur_bal": "Average balance per account",
    "tot_cur_bal": "Total current balance",
    "total_bal_ex_mort": "Total balance excluding mortgage",
    "num_actv_rev_tl": "Number of active revolving accounts",
    "open_acc": "Number of open accounts",
    "total_acc": "Total number of credit accounts",
}


def reason_label(feature: str) -> str:
    return REASON_LABELS.get(feature, feature.replace("_", " ").capitalize())
