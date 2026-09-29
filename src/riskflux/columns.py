"""Column registry: the code version of docs/COLUMN_AUDIT.md.

Every column the pipeline uses is declared here once, by role. Model code selects
inputs only through FEATURES, so leakage columns cannot reach the model by accident.
"""

# --- keys, label, and raw columns used only to define the population -------------
ID = "id"
ISSUE_DATE = "issue_d"
TARGET = "y"  # 1 = Charged Off / Default, 0 = Fully Paid

TERM = "term"
APPLICATION_TYPE = "application_type"
LOAN_STATUS = "loan_status"

GOOD_STATUSES = frozenset({"Fully Paid"})
BAD_STATUSES = frozenset({"Charged Off", "Default"})

# --- model features: known at the moment of the credit decision ------------------
CATEGORICAL_FEATURES = ["home_ownership", "purpose"]
DATE_FEATURES = ["earliest_cr_line"]
NUMERIC_FEATURES = [
    # application
    "loan_amnt", "emp_length", "annual_inc", "dti",
    # credit bureau, available since 2007
    "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "acc_now_delinq", "chargeoff_within_12_mths",
    "delinq_amnt", "pub_rec_bankruptcies", "tax_liens",
    # credit bureau wave 2, available since 2012-09
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
    "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
    "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit",
]  # fmt: skip
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES

# --- auxiliary columns: never model inputs, stored in a separate file -------------
# Lending Club's own underwriting outputs (D4): cost function + grade ablation only.
LC_DECISION_NUMERIC = ["int_rate", "installment", "funded_amnt", "funded_amnt_inv"]
LC_DECISION_CATEGORICAL = ["grade", "sub_grade", "initial_list_status", "verification_status"]
# Post-origination outcomes: used only to estimate LGD and the prepayment haircut.
OUTCOME = ["total_rec_prncp", "total_rec_int", "recoveries", "collection_recovery_fee"]
# Proxies for protected attributes: fairness slicing only.
FAIRNESS = ["addr_state", "zip_code"]

AUX_NUMERIC = LC_DECISION_NUMERIC + OUTCOME
AUX_CATEGORICAL = LC_DECISION_CATEGORICAL + FAIRNESS
AUX = AUX_NUMERIC + AUX_CATEGORICAL

# --- known leakage: information that only exists after the loan was issued --------
LEAKAGE = [
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "pymnt_plan",
    "hardship_flag", "debt_settlement_flag",
]  # fmt: skip
LEAKAGE_PREFIXES = ("hardship_",)

PERCENT_COLUMNS = frozenset({"revol_util", "int_rate"})  # stored as strings like "83.7%"

# Everything the clean stage reads from the raw parquet
RAW_COLUMNS = [ID, ISSUE_DATE, TERM, APPLICATION_TYPE, LOAN_STATUS, *FEATURES, *AUX]
