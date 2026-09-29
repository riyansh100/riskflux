import pickle

import pandas as pd
import pytest

from riskflux import columns as C
from riskflux.data.clean import clean
from riskflux.features.engineer import MODEL_FEATURES, engineer
from riskflux.features.pipeline import CategoricalVocab, build_feature_pipeline
from tests.conftest import make_raw


@pytest.fixture
def loans(population) -> pd.DataFrame:
    loans, _ = clean(
        make_raw(
            {C.ISSUE_DATE: "Mar-2014", "earliest_cr_line": "Nov-2009", "home_ownership": "RENT"},
            {C.ISSUE_DATE: "Jan-2016", "earliest_cr_line": "Jan-1990", "home_ownership": "OWN"},
        ),
        population,
    )
    return loans


def test_derived_features(loans):
    out = engineer(loans)
    first = out.iloc[0]
    assert first["fico_mid"] == 702.0  # (700 + 704) / 2
    assert first["credit_history_months"] == 52.0  # Nov-2009 -> Mar-2014
    assert first["loan_to_income"] == pytest.approx(10_000 / 60_000)
    assert out["credit_history_months"].iloc[1] == 312.0  # 26 years


def test_output_is_exactly_model_features_in_order(loans):
    """No id, no label, no issue_d, no raw fico range: only declared model inputs."""
    assert list(engineer(loans).columns) == MODEL_FEATURES


def test_missing_input_column_fails_loudly(loans):
    with pytest.raises(KeyError, match="earliest_cr_line"):
        engineer(loans.drop(columns="earliest_cr_line"))


def test_unseen_category_becomes_missing_not_an_error():
    vocab = CategoricalVocab(columns=["home_ownership"]).fit(
        pd.DataFrame({"home_ownership": ["RENT", "OWN"]})
    )
    out = vocab.transform(pd.DataFrame({"home_ownership": ["OWN", "MORTGAGE"]}))
    assert out["home_ownership"].iloc[0] == "OWN"
    assert pd.isna(out["home_ownership"].iloc[1])
    assert list(out["home_ownership"].cat.categories) == ["OWN", "RENT"]


def test_category_codes_are_stable_regardless_of_input_order(loans):
    """Same category -> same integer code, whatever order serving rows arrive in."""
    pipe = build_feature_pipeline().fit(loans)
    forward = pipe.transform(loans)["home_ownership"].cat.codes.tolist()
    backward = pipe.transform(loans.iloc[::-1])["home_ownership"].cat.codes.tolist()
    assert forward == backward[::-1]


def test_fitted_pipeline_survives_pickling(loans):
    """MLflow pickles the pipeline; the reloaded copy must transform identically."""
    pipe = build_feature_pipeline().fit(loans)
    reloaded = pickle.loads(pickle.dumps(pipe))
    pd.testing.assert_frame_equal(pipe.transform(loans), reloaded.transform(loans))
