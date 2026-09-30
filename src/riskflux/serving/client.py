"""Turning stored loans into API requests: used by parity tests, the latency benchmark,
and (Step 9) replaying later loans through the API as simulated production traffic."""

import math

import pandas as pd

from riskflux import columns as C


def row_to_request(row: pd.Series) -> dict:
    """A clean-stage row -> the JSON body an API client would send for that loan.
    NaN becomes null (missing), dates become ISO strings, issue_d is the application date."""
    body = {"application_date": row[C.ISSUE_DATE].date().isoformat()}
    for name in C.FEATURES:
        value = row[name]
        if isinstance(value, pd.Timestamp):
            body[name] = value.date().isoformat()
        elif isinstance(value, float) and math.isnan(value):
            body[name] = None
        else:
            body[name] = value
    return body


def frame_to_requests(loans: pd.DataFrame) -> list[dict]:
    return [row_to_request(row) for _, row in loans.iterrows()]
