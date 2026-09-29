"""File paths and pipeline parameters (params.yaml), loaded and checked in one place."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARAMS_PATH = Path("params.yaml")

RAW_ZIP = Path("data/raw/lending-club-20072020q1.zip")
RAW_PARQUET = Path("data/interim/loans_raw.parquet")
CLEAN_PARQUET = Path("data/interim/loans_clean.parquet")
AUX_PARQUET = Path("data/interim/loans_aux.parquet")
SPLIT_DIR = Path("data/processed")
SPLIT_STATS = Path("reports/split_stats.json")  # DVC metrics file, committed to git

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"  # fitted pipeline: features -> LightGBM -> calibration
POLICY_PATH = MODEL_DIR / "policy.json"  # decision threshold + cost parameters
TRAIN_METRICS = Path("reports/train_metrics.json")
TEST_METRICS = Path("reports/test_metrics.json")
CALIBRATION_CURVE = Path("reports/calibration_test.csv")
ABLATION_METRICS = Path("reports/ablation.json")

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Window:
    """Inclusive range of issue months. Both ends are month starts (e.g. 2015-12-01)."""

    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_months(cls, start: str, end: str) -> "Window":
        window = cls(pd.Timestamp(start), pd.Timestamp(end))
        if window.start > window.end:
            raise ValueError(f"window starts after it ends: {start} > {end}")
        return window

    def contains(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.start, self.end)


@dataclass(frozen=True)
class PopulationParams:
    term_months: int
    application_type: str
    window: Window


@dataclass(frozen=True)
class ModelParams:
    seed: int
    early_stopping_rounds: int
    lightgbm: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CostParams:
    threshold_grid: np.ndarray  # candidate PD cut-offs for the cost-optimal threshold


@dataclass(frozen=True)
class Params:
    population: PopulationParams
    splits: dict[str, Window]
    model: ModelParams
    cost: CostParams


def load_params(path: Path = PARAMS_PATH) -> Params:
    raw = yaml.safe_load(Path(path).read_text())
    pop = raw["population"]
    population = PopulationParams(
        term_months=int(pop["term_months"]),
        application_type=pop["application_type"],
        window=Window.from_months(pop["issue_start"], pop["issue_end"]),
    )
    splits = {name: Window.from_months(**raw["split"][name]) for name in raw["split"]}
    check_splits(population.window, splits)

    grid = raw["cost"]["threshold_grid"]
    cost = CostParams(
        threshold_grid=np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 6)
    )
    return Params(
        population=population, splits=splits, model=ModelParams(**raw["model"]), cost=cost
    )


def check_splits(population: Window, splits: dict[str, Window]) -> None:
    """Splits must be train/val/test in time order, with no gaps or overlaps,
    and together cover exactly the population window."""
    if tuple(splits) != SPLIT_NAMES:
        raise ValueError(f"splits must be {SPLIT_NAMES} in that order, got {tuple(splits)}")
    windows = list(splits.values())
    if windows[0].start != population.start or windows[-1].end != population.end:
        raise ValueError("splits must start and end exactly at the population window")
    for earlier, later in zip(windows, windows[1:], strict=False):
        if later.start != earlier.end + pd.DateOffset(months=1):
            raise ValueError(f"split windows are not contiguous: {earlier} -> {later}")
