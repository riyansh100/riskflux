"""Train/serve parity: the same loan must produce identical model input whether it
arrives as a row of the training data or as a JSON request to the API.

    batch path:    clean-stage frame ───────────────────────────────────┐
                                                                       ├─> fitted pipeline
    serving path:  JSON -> LoanApplication -> applications_to_frame ───┘

Step 6 extends this to model scores (same features + same model -> same score).
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.data.clean import clean
from riskflux.features.pipeline import build_feature_pipeline
from riskflux.schemas import LoanApplication, applications_to_frame
from tests.conftest import make_raw

TEST_SPLIT = Path(__file__).parents[2] / "data/processed/test.parquet"


def to_request_json(row: pd.Series) -> str:
    """Serialize a training row the way an API client would send it."""
    body = {"application_date": row[C.ISSUE_DATE].date().isoformat()}
    for name in C.FEATURES:
        value = row[name]
        if isinstance(value, pd.Timestamp):
            body[name] = value.date().isoformat()
        elif isinstance(value, float) and math.isnan(value):
            body[name] = None
        else:
            body[name] = value
    return json.dumps(body)


def serving_features(loans: pd.DataFrame, pipeline) -> pd.DataFrame:
    bodies = [to_request_json(row) for _, row in loans.iterrows()]
    requests = [LoanApplication.model_validate_json(body) for body in bodies]
    return pipeline.transform(applications_to_frame(requests))


def assert_parity(loans: pd.DataFrame) -> None:
    loans = loans.reset_index(drop=True)
    pipeline = build_feature_pipeline().fit(loans)
    pd.testing.assert_frame_equal(pipeline.transform(loans), serving_features(loans, pipeline))


def test_parity_on_synthetic_loans(population):
    loans, _ = clean(
        make_raw(
            {},
            {"emp_length": None, "mths_since_last_delinq": None, "dti": None},  # missing values
            {"home_ownership": "OWN", "purpose": "car", "emp_length": "10+ years"},
            {"earliest_cr_line": "Jan-1970", C.ISSUE_DATE: "Apr-2017"},
        ),
        population,
    )
    assert_parity(loans)


@pytest.mark.skipif(not TEST_SPLIT.exists(), reason="needs data/processed (run the pipeline)")
def test_parity_on_real_loans():
    loans = pd.read_parquet(TEST_SPLIT).sample(n=2_000, random_state=0)
    assert_parity(loans)
