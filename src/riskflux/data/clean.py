"""Clean stage: raw all-string parquet -> typed, labeled population.

Outputs two files so leakage is prevented by construction:
- loans_clean.parquet: id, issue_d, y, and the model FEATURES. Nothing else.
- loans_aux.parquet:   id + LC decision, outcome, and fairness columns
                       (for the cost function, ablation, and fairness slicing).

Rules:
- Stateless only. Nothing here is learned from the data (no quantiles, no imputation).
  Anything learned must be fitted on the train split in the feature pipeline,
  otherwise test-set information leaks into training.
- Strict parsing. An unexpected value raises instead of silently becoming NaN, so
  a change in the source data fails loudly.
- Runs in batches because the raw parquet (2.9M rows) doesn't fit in 8 GB RAM as
  strings. Every operation is row-wise, so batching gives the same result.

Run: uv run python -m riskflux.data.clean
"""

import pandas as pd
import pyarrow.parquet as pq

from riskflux import columns as C
from riskflux.config import AUX_PARQUET, CLEAN_PARQUET, RAW_PARQUET, PopulationParams, load_params
from riskflux.schemas import AUX_SCHEMA, CLEAN_SCHEMA

EMP_LENGTH_YEARS = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    **{f"{n} years": float(n) for n in range(2, 10)},
    "10+ years": 10.0,
}
BATCH_ROWS = 250_000


def _raise_unparsed(raw: pd.Series, parsed: pd.Series, what: str) -> None:
    bad = raw.notna() & parsed.isna()
    if bad.any():
        examples = raw[bad].unique()[:5].tolist()
        raise ValueError(f"{raw.name}: {bad.sum()} values are not {what}, e.g. {examples}")


def parse_numeric(raw: pd.Series) -> pd.Series:
    text = raw.str.strip()
    if raw.name in C.PERCENT_COLUMNS:
        text = text.str.rstrip("%")
    parsed = pd.to_numeric(text, errors="coerce").astype("float64")
    _raise_unparsed(raw, parsed, "numeric")
    return parsed


def parse_month(raw: pd.Series) -> pd.Series:
    """'Dec-2011' -> Timestamp('2011-12-01')."""
    parsed = pd.to_datetime(raw, format="%b-%Y", errors="coerce")
    _raise_unparsed(raw, parsed, "Mon-YYYY dates")
    return parsed.astype("datetime64[ns]")  # fixed unit: pandas 3 defaults vary by input


def parse_emp_length(raw: pd.Series) -> pd.Series:
    """'10+ years' -> 10.0, '< 1 year' -> 0.0. Missing stays NaN (missing is informative)."""
    parsed = raw.map(EMP_LENGTH_YEARS).astype("float64")
    _raise_unparsed(raw, parsed, "known employment lengths")
    return parsed


def population_mask(
    raw: pd.DataFrame, issue: pd.Series, pop: PopulationParams, require_outcome: bool = True
) -> pd.Series:
    mask = (
        (raw[C.TERM].str.strip() == f"{pop.term_months} months")
        & (raw[C.APPLICATION_TYPE] == pop.application_type)
        & pop.window.contains(issue)
    )
    if require_outcome:
        mask &= raw[C.LOAN_STATUS].isin(C.GOOD_STATUSES | C.BAD_STATUSES)
    return mask


def clean(
    raw: pd.DataFrame, pop: PopulationParams, require_outcome: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (clean, aux) for the rows of `raw` that belong to the population.

    require_outcome=False keeps loans still in flight (for replayed production traffic):
    their label is missing (<NA>) instead of the row being dropped."""
    # Junk rows: section headers left over from LC's concatenated files.
    raw = raw[raw[C.ID].str.fullmatch(r"\d+", na=False)]

    issue = parse_month(raw[C.ISSUE_DATE])
    keep = population_mask(raw, issue, pop, require_outcome)
    raw, issue = raw[keep], issue[keep]

    out = pd.DataFrame(index=raw.index)
    out[C.ID] = raw[C.ID].astype("int64")
    out[C.ISSUE_DATE] = issue
    status = raw[C.LOAN_STATUS]
    if require_outcome:
        out[C.TARGET] = status.isin(C.BAD_STATUSES).astype("int8")
    else:
        label = pd.Series(pd.NA, index=raw.index, dtype="Int8")
        label[status.isin(C.BAD_STATUSES)] = 1
        label[status.isin(C.GOOD_STATUSES)] = 0
        out[C.TARGET] = label

    for col in C.NUMERIC_FEATURES:
        out[col] = parse_emp_length(raw[col]) if col == "emp_length" else parse_numeric(raw[col])
    for col in C.CATEGORICAL_FEATURES:
        out[col] = raw[col].str.strip()
    for col in C.DATE_FEATURES:
        out[col] = parse_month(raw[col])

    # Impossible value: debt-to-income can't be negative (1 row in the population).
    out.loc[out["dti"] < 0, "dti"] = float("nan")

    aux = pd.DataFrame(index=raw.index)
    aux[C.ID] = out[C.ID]
    for col in C.AUX_NUMERIC:
        aux[col] = parse_numeric(raw[col])
    for col in C.AUX_CATEGORICAL:
        aux[col] = raw[col].str.strip()

    return out.reset_index(drop=True), aux.reset_index(drop=True)


def clean_raw_file(
    pop: PopulationParams, require_outcome: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the raw parquet in batches through clean() (the full file doesn't fit in RAM)."""
    batches = pq.ParquetFile(RAW_PARQUET).iter_batches(batch_size=BATCH_ROWS, columns=C.RAW_COLUMNS)
    parts = [clean(batch.to_pandas(), pop, require_outcome) for batch in batches]
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


def main() -> None:
    params = load_params()
    loans, aux = clean_raw_file(params.population)
    loans, aux = CLEAN_SCHEMA.validate(loans), AUX_SCHEMA.validate(aux)

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(CLEAN_PARQUET, index=False)
    aux.to_parquet(AUX_PARQUET, index=False)
    rate = loans[C.TARGET].mean()
    print(f"clean: {len(loans):,} loans, default rate {rate:.2%} -> {CLEAN_PARQUET}")
    print(f"aux:   {len(aux):,} rows -> {AUX_PARQUET}")


if __name__ == "__main__":
    main()
