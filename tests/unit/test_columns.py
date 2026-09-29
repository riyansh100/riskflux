"""Guards on the column registry: leakage can't reach the model through a typo or copy-paste."""

import re
from pathlib import Path

from riskflux import columns as C

AUDIT_DOC = Path(__file__).parents[2] / "docs/COLUMN_AUDIT.md"


def test_no_duplicates_within_or_across_roles():
    declared = [C.ID, C.ISSUE_DATE, C.TERM, C.APPLICATION_TYPE, C.LOAN_STATUS, *C.FEATURES, *C.AUX]
    assert len(declared) == len(set(declared))


def test_features_contain_no_leakage_or_lc_decisions():
    features = set(C.FEATURES)
    assert features.isdisjoint(C.LEAKAGE)
    assert not any(f.startswith(C.LEAKAGE_PREFIXES) for f in features)
    assert features.isdisjoint(C.LC_DECISION_NUMERIC + C.LC_DECISION_CATEGORICAL)
    assert features.isdisjoint(C.FAIRNESS)


def test_outcome_columns_are_flagged_as_leakage():
    assert set(C.OUTCOME) <= set(C.LEAKAGE)


def test_features_match_the_eda_audit():
    """Code and docs must agree. If this fails, update docs/COLUMN_AUDIT.md (re-run the
    EDA notebook) or columns.py so they describe the same feature set."""
    section = AUDIT_DOC.read_text().split("## candidate_feature")[1].split("\n## ")[0]
    documented = set(re.findall(r"`([a-z0-9_]+)`", section))
    assert documented == set(C.FEATURES)
    assert len(C.FEATURES) == 61
