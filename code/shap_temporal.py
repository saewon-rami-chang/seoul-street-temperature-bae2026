"""SHAP values of the temporal model (run train_temporal.py first)."""

import os
os.environ["OMP_NUM_THREADS"] = "8"

import gc
import json
import logging
import numpy as np
import pandas as pd
import polars as pl
import seoul_data
import xgboost as xgb
import shap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"serial_no", "datetime", "lat", "lon", "aws_temp_interp", "target_resid"}


def main():
    model_dir = (seoul_data.default_results_dir() / "Full_Model").resolve()
    cache_dir = seoul_data.default_data_dir()
    shap_dir = model_dir / "SHAP"
    min_year = 2023

    if (shap_dir / "SHAP_Full.csv").exists() and (shap_dir / "SHAP_Values_Full.parquet").exists():
        logger.info("SHAP already complete. Delete Full_Model/SHAP/ to rerun.")
        return

    model_path = model_dir / "Full_Model.json"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run train_full_model.py first.")
        return
    model = xgb.Booster()
    model.load_model(str(model_path))
    logger.info(f"Loaded {model_path}")

    files = [u for u in seoul_data.units(cache_dir) if u.year >= min_year]
    schema_cols = seoul_data.schema_names(files)
    features = sorted(schema_cols - META_COLS)
    logger.info(f"Features: {len(features)}, Chunks: {len(files)}")

    rng = np.random.RandomState(42)
    idxs = rng.choice(len(files), size=min(3, len(files)), replace=False)
    frames = [files[i].read().select(features) for i in idxs]
    combined = pl.concat(frames)
    X = combined.sample(n=min(50_000, combined.height), seed=42).to_pandas()
    del frames, combined; gc.collect()
    logger.info(f"Sample: {len(X):,} rows")

    shap_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Computing SHAP...")
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)

    mabs = np.abs(sv).mean(axis=0)
    imp = dict(sorted({f: float(v) for f, v in zip(features, mabs)}.items(),
                       key=lambda x: x[1], reverse=True))

    with open(shap_dir / "SHAP_Full.json", "w") as f:
        json.dump(imp, f, indent=4)
    pd.DataFrame(list(imp.items()), columns=["Feature", "Mean_Abs_SHAP"]).to_csv(
        shap_dir / "SHAP_Full.csv", index=False)
    pd.DataFrame(sv, columns=features).to_parquet(
        shap_dir / "SHAP_Values_Full.parquet", index=False)
    pd.DataFrame(X).to_parquet(
        shap_dir / "SHAP_Features_Full.parquet", index=False)

    logger.info(f"Top 5: {list(imp.keys())[:5]}")
    logger.info(f"Done — {shap_dir}/")


if __name__ == "__main__":
    main()