"""Split stage: clean population -> out-of-time train / val / test.

Splits by issue month, never at random: the model is always evaluated on loans
issued after the ones it was trained on, just like in production. A random split
would let the model see the future and hide the drift in the data.

Run: uv run python -m riskflux.data.split
"""

import json

import pandas as pd

from riskflux import columns as C
from riskflux.config import CLEAN_PARQUET, SPLIT_DIR, SPLIT_STATS, Window, load_params
from riskflux.schemas import CLEAN_SCHEMA


def split(loans: pd.DataFrame, windows: dict[str, Window]) -> dict[str, pd.DataFrame]:
    parts = {
        name: loans[window.contains(loans[C.ISSUE_DATE])].reset_index(drop=True)
        for name, window in windows.items()
    }
    # Every loan must land in exactly one split: no gaps, no overlaps.
    assigned = sum(len(part) for part in parts.values())
    if assigned != len(loans):
        raise ValueError(f"{assigned:,} rows assigned to splits, expected {len(loans):,}")
    return parts


def main() -> None:
    params = load_params()
    loans = pd.read_parquet(CLEAN_PARQUET)
    parts = split(loans, params.splits)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in parts.items():
        CLEAN_SCHEMA.validate(part)
        part.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        stats[name] = {"loans": len(part), "default_rate": round(part[C.TARGET].mean(), 5)}
        dates = part[C.ISSUE_DATE]
        print(
            f"{name:5s} {dates.min():%Y-%m} .. {dates.max():%Y-%m}  "
            f"{len(part):>7,} loans  default rate {part[C.TARGET].mean():.2%}"
        )

    # Tracked as DVC metrics: `dvc metrics diff` shows how the data changed between commits.
    SPLIT_STATS.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_STATS.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
