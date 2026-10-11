"""Reads the released yearly Parquet files in monthly batches, with the column names used by the training code."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

RENAME = {"sensor_id": "serial_no", "timestamp": "datetime", "aws_baseline": "aws_temp_interp"}
EXTRA = ["hardware_generation", "loro_region", "sdot_temp"]
FILE_RE = re.compile(r"seoul_sdot_features_(\d{4})\.parquet$")


def default_data_dir() -> Path:
    """$SEOUL_DATA_DIR if set, else <repository>/data."""
    env = os.environ.get("SEOUL_DATA_DIR")
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Unit:
    """One monthly row group of a yearly file."""
    path: Path
    rg: int
    year: int

    def read(self) -> pl.DataFrame:
        t = pq.ParquetFile(self.path).read_row_group(self.rg)
        return pl.from_arrow(t).rename(RENAME).drop(EXTRA)


def units(data_dir: Path, min_year: int = 2023, max_year: int = 9999) -> list[Unit]:
    files = sorted(p for p in Path(data_dir).glob("seoul_sdot_features_*.parquet") if FILE_RE.search(p.name))
    if not files:
        raise FileNotFoundError(f"No seoul_sdot_features_<year>.parquet files in {data_dir} "
                                f"(download the dataset or set SEOUL_DATA_DIR)")
    out = []
    for p in files:
        y = int(FILE_RE.search(p.name).group(1))
        if min_year <= y <= max_year:
            out += [Unit(p, i, y) for i in range(pq.ParquetFile(p).num_row_groups)]
    return out


def scan(us: list[Unit]) -> pl.LazyFrame:
    """Lazy frame over the years covered by the units (units always cover whole years)."""
    paths = sorted({u.path for u in us})
    return pl.scan_parquet(paths).rename(RENAME).drop(EXTRA)


def schema_names(us: list[Unit]) -> set[str]:
    return set(scan(us[:1]).collect_schema().names())


def default_results_dir() -> Path:
    """$SEOUL_RESULTS_DIR if set, else <repository>/results (all training scripts write here)."""
    env = os.environ.get("SEOUL_RESULTS_DIR")
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[1] / "results"
