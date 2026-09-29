"""File paths and pipeline parameters (params.yaml), loaded and checked in one place."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

PARAMS_PATH = Path("params.yaml")

RAW_ZIP = Path("data/raw/lending-club-20072020q1.zip")
RAW_PARQUET = Path("data/interim/loans_raw.parquet")
CLEAN_PARQUET = Path("data/interim/loans_clean.parquet")
AUX_PARQUET = Path("data/interim/loans_aux.parquet")
SPLIT_DIR = Path("data/processed")

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
class Params:
    population: PopulationParams
    splits: dict[str, Window]


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
    return Params(population=population, splits=splits)


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
