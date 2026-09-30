"""Latency benchmark for a running RiskFlux API (sequential single-loan requests).

Sends real test-split loans as JSON, one request at a time, and reports p50/p95/p99
latency measured by the client (network + validation + features + model + SHAP).

Usage: uv run python scripts/benchmark_api.py --url http://localhost:8080 --n 500
"""

import argparse
import statistics
import time

import httpx
import pandas as pd

from riskflux.config import SPLIT_DIR
from riskflux.serving.client import frame_to_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    loans = pd.read_parquet(SPLIT_DIR / "test.parquet").sample(args.n, random_state=0)
    bodies = frame_to_requests(loans)

    latencies, decisions = [], {}
    with httpx.Client(base_url=args.url, timeout=10) as client:
        client.post("/predict", json=bodies[0])  # warm-up
        for body in bodies:
            start = time.perf_counter()
            response = client.post("/predict", json=body)
            latencies.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
            decision = response.json()["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1

    q = statistics.quantiles(latencies, n=100)
    print(f"{args.n} sequential requests to {args.url}")
    print(
        f"latency ms: p50={q[49]:.1f}  p95={q[94]:.1f}  p99={q[98]:.1f}  max={max(latencies):.1f}"
    )
    rps = 1000 / statistics.mean(latencies)
    print(f"throughput (1 client): {rps:.0f} req/s | decisions: {decisions}")


if __name__ == "__main__":
    main()
