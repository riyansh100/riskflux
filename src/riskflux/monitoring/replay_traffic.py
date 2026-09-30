"""Replay later loans through a RUNNING API, in issue-date order, as production traffic.

The API writes what it served to its prediction log (PREDICTION_LOG_PATH); the drift
job then reads that log, exactly as it would read real production logs.

Start the API with a prediction log, then replay:
    PREDICTION_LOG_PATH=data/monitoring/predictions.jsonl \
        uv run uvicorn riskflux.serving.app:app --port 8080
    uv run python -m riskflux.monitoring.replay_traffic --url http://localhost:8080
"""

import argparse
import time

import httpx
import pandas as pd

from riskflux import columns as C
from riskflux.config import REPLAY_PARQUET
from riskflux.serving.client import frame_to_requests


def replay(url: str, loans: pd.DataFrame, batch_size: int = 500) -> dict[str, int]:
    decisions: dict[str, int] = {}
    with httpx.Client(base_url=url, timeout=120) as client:
        client.get("/ready").raise_for_status()
        for month, month_loans in loans.groupby(loans[C.ISSUE_DATE].dt.to_period("M")):
            start = time.perf_counter()
            for i in range(0, len(month_loans), batch_size):
                chunk = frame_to_requests(month_loans.iloc[i : i + batch_size])
                response = client.post("/predict/batch", json={"applications": chunk})
                response.raise_for_status()
                for p in response.json()["predictions"]:
                    decisions[p["decision"]] = decisions.get(p["decision"], 0) + 1
            print(f"{month}: {len(month_loans):>5,} loans in {time.perf_counter() - start:.1f}s")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    loans = pd.read_parquet(REPLAY_PARQUET)
    decisions = replay(args.url, loans, args.batch_size)
    print(f"replayed {len(loans):,} loans -> decisions {decisions}")


if __name__ == "__main__":
    main()
