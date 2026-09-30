"""Replay stage: loans issued AFTER the modeling window -> simulated production traffic.

Same product (36-month, individual) and the same parsing code as the clean stage, but
loans still in flight are kept (label <NA>): production traffic doesn't come with
outcomes. A fixed-seed sample per issue month keeps a replay to minutes and makes it
reproducible.

Run: uv run python -m riskflux.data.replay
"""

import pandas as pd

from riskflux import columns as C
from riskflux.config import REPLAY_PARQUET, PopulationParams, load_params
from riskflux.data.clean import clean_raw_file
from riskflux.schemas import REPLAY_SCHEMA


def sample_by_month(loans: pd.DataFrame, per_month: int, seed: int) -> pd.DataFrame:
    """Up to `per_month` loans from each issue month, in issue-date order."""
    shuffled = loans.sample(frac=1.0, random_state=seed)  # seeded shuffle, then first N per month
    sampled = shuffled.groupby(C.ISSUE_DATE).head(per_month)
    return sampled.sort_values([C.ISSUE_DATE, C.ID]).reset_index(drop=True)


def main() -> None:
    params = load_params()
    replay_population = PopulationParams(
        term_months=params.population.term_months,
        application_type=params.population.application_type,
        window=params.monitoring.replay,
    )
    loans, _ = clean_raw_file(replay_population, require_outcome=False)
    replay = REPLAY_SCHEMA.validate(
        sample_by_month(loans, params.monitoring.sample_per_month, params.monitoring.seed)
    )

    REPLAY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(REPLAY_PARQUET, index=False)
    months = replay[C.ISSUE_DATE].dt.to_period("M")
    print(
        f"replay: {len(replay):,} loans ({len(loans):,} available), {months.nunique()} months "
        f"{months.min()}..{months.max()}, {replay[C.TARGET].isna().mean():.0%} without outcome "
        f"-> {REPLAY_PARQUET}"
    )


if __name__ == "__main__":
    main()
