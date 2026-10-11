"""Shared paths, metrics and the sensor bootstrap."""

from pathlib import Path

import os
import sys

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # code/
import seoul_data  # noqa: E402

TRAIN = seoul_data.default_results_dir()   # training outputs: results/<run>/ (written by code/train_*.py)
# inputs that are not released (see README)
SEOUL = Path(os.environ.get("SEOUL_EXTRA_INPUTS", HERE.parents[1] / "extra_inputs")).expanduser().resolve()
NUMBERS = HERE.parents[1] / "tables"
(NUMBERS / "figures").mkdir(parents=True, exist_ok=True)


def cache_units():
    """Monthly units of the released feature table (internal column names)."""
    return seoul_data.units(seoul_data.default_data_dir())


def scan_cache():
    return seoul_data.scan(cache_units())
NUMBERS.mkdir(exist_ok=True)

PRED = {  # validation scheme -> predictions of the published models
    "Random 80/20": TRAIN / "Random8020_Outputs/Predictions/Predictions.parquet",
    "Spatial LORO": TRAIN / "Discovery_Outputs/Predictions/Predictions.parquet",
    "Temporal 2025": TRAIN / "Full_Model/Temporal_Holdout/Predictions_2025.parquet",
}
REGION_LABEL = {
    "A_Mapo": "A: Mapo", "B_Gangnam": "B: Gangnam", "C_Central": "C: Central",
    "D_Gangdong": "D: Gangdong", "E_Dobong": "E: Dobong", "F_Gwanak": "F: Gwanak",
}
MONTH_TO_SEASON = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
                   3: "Spring (MAM)", 4: "Spring (MAM)", 5: "Spring (MAM)",
                   6: "Summer (JJA)", 7: "Summer (JJA)", 8: "Summer (JJA)",
                   9: "Autumn (SON)", 10: "Autumn (SON)", 11: "Autumn (SON)"}
N_BOOT = 1000
SEED = 42


def load_pred(path, extra=()):
    """Predictions as pandas; polars reader (pyarrow 19 fails on these files)."""
    cols = ["datetime", "serial_no", "actual_temp", "pred_temp", "aws_temp_interp",
            "target_resid", "pred_resid", *extra]
    df = pl.read_parquet(path, columns=list(dict.fromkeys(cols))).to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, np.float64); y_pred = np.asarray(y_pred, np.float64)
    e = y_pred - y_true
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    return {"N_obs": len(e), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "MBE": float(np.mean(e)), "R2": float(1 - np.sum(e ** 2) / ss_t) if ss_t > 0 else np.nan}


def sensor_ci(df, pred_col="pred_temp", true_col="actual_temp", strata=None, rng=None) -> dict:
    """95 % CI of RMSE/MAE/MBE resampling sensors with replacement (within strata if given)."""
    rng = rng or np.random.default_rng(SEED)
    e = df[pred_col].to_numpy(np.float64) - df[true_col].to_numpy(np.float64)
    g = pd.DataFrame({"s": df["serial_no"].to_numpy(), "k": df[strata].to_numpy() if strata else 0,
                      "n": 1.0, "e": e, "e2": e ** 2, "ae": np.abs(e)})
    sums = g.groupby(["k", "s"], sort=False)[["n", "e", "e2", "ae"]].sum()
    tot = np.zeros((N_BOOT, 4))
    for _, blk in sums.groupby(level="k", sort=False):
        a = blk.to_numpy()
        w = rng.multinomial(len(a), np.full(len(a), 1 / len(a)), size=N_BOOT)
        tot += w @ a
    n = tot[:, 0]
    out = {}
    for name, v in (("RMSE", np.sqrt(tot[:, 2] / n)), ("MAE", tot[:, 3] / n), ("MBE", tot[:, 1] / n)):
        out[f"{name}_CI_lo"], out[f"{name}_CI_hi"] = np.percentile(v, [2.5, 97.5])
    out["N_sensors"] = int(sums.shape[0])
    return out


def save(df: pd.DataFrame, name: str, floatfmt="%.4f"):
    path = NUMBERS / name
    df.to_csv(path, index=False, float_format=floatfmt)
    print(f"  -> {path.relative_to(HERE.parents[1])}")
    return path
