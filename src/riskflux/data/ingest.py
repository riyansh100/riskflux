"""Ingest stage: raw Kaggle zip -> single parquet file.

Design:
- Streams the CSV in blocks. The file is 1.77 GB; loading it into pandas at once
  needs ~6-10 GB RAM, more than a laptop can spare.
- Reads EVERY column as a string. Type inference from the first block is wrong for
  columns that are empty in early (2007-2012) loans, and a wrong guess would crash
  or silently corrupt later blocks. Parsing numbers/dates/percents happens in the
  clean stage, where it is explicit and tested.
- Drops the unnamed leading column (a leftover pandas index, not data).

Run: uv run python -m riskflux.data.ingest
"""

import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

RAW_ZIP = Path("data/raw/lending-club-20072020q1.zip")
CSV_MEMBER = "Loan_status_2007-2020Q3.gzip"  # an UNCOMPRESSED csv despite the extension
OUT_PATH = Path("data/interim/loans_raw.parquet")

INDEX_COL = "_pandas_index"
BLOCK_SIZE = 64 << 20  # 64 MB per streamed block


def read_header(zf: zipfile.ZipFile) -> list[str]:
    with zf.open(CSV_MEMBER) as f:
        names = f.readline().decode("utf-8").rstrip("\r\n").split(",")
    names[0] = names[0] or INDEX_COL  # the leading column has an empty name
    return names


def ingest(raw_zip: Path = RAW_ZIP, out_path: Path = OUT_PATH) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(raw_zip) as zf:
        names = read_header(zf)
        keep = [n for n in names if n != INDEX_COL]

        read_opts = pv.ReadOptions(column_names=names, skip_rows=1, block_size=BLOCK_SIZE)
        convert_opts = pv.ConvertOptions(
            column_types={n: pa.string() for n in names},
            include_columns=keep,
            strings_can_be_null=True,  # empty cells -> null, not ""
        )

        rows = 0
        with zf.open(CSV_MEMBER) as f:
            reader = pv.open_csv(f, read_options=read_opts, convert_options=convert_opts)
            with pq.ParquetWriter(out_path, reader.schema, compression="zstd") as writer:
                for batch in reader:
                    writer.write_batch(batch)
                    rows += batch.num_rows
    return rows


if __name__ == "__main__":
    n = ingest()
    print(f"wrote {n:,} rows -> {OUT_PATH}")
