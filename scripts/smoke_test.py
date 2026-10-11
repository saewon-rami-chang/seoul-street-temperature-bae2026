"""Setup check: random holdout on 1 % of 2023, prints RMSE. Not a paper result."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import seoul_data  # noqa: E402

PARAMS = {"objective": "reg:absoluteerror", "eval_metric": "mae", "tree_method": "hist", "max_bin": 256,
          "learning_rate": 0.05, "max_depth": 12, "min_child_weight": 50, "subsample": 0.9,
          "colsample_bytree": 0.55, "gamma": 0.7, "seed": 42}
ROUNDS = 300


def main():
    t0 = time.time()
    units = seoul_data.units(seoul_data.default_data_dir(), 2023, 2023)
    df = (seoul_data.scan(units).filter(pl.col("target_resid").abs() < 25)
          .collect().sample(fraction=0.01, seed=42))
    feats = sorted(set(df.columns) - {"serial_no", "datetime", "lat", "lon", "aws_temp_interp", "target_resid"})
    test_ids = set((ROOT / "code/configs/random8020_test_sensors.txt").read_text().split())
    test = df.filter(pl.col("serial_no").is_in(list(test_ids))); train = df.filter(~pl.col("serial_no").is_in(list(test_ids)))
    print(f"2023 1% sample: {df.height:,} rows, {len(feats)} features | train {train.height:,} rows "
          f"({train['serial_no'].n_unique()} sensors), test {test.height:,} rows ({test['serial_no'].n_unique()} sensors)")
    d = xgb.DMatrix(train.select(feats).to_numpy().astype(np.float32), label=train["target_resid"].to_numpy())
    model = xgb.train(PARAMS, d, num_boost_round=ROUNDS)
    pred = model.predict(xgb.DMatrix(test.select(feats).to_numpy().astype(np.float32)))
    obs = (test["aws_temp_interp"] + test["target_resid"]).to_numpy()
    aws = test["aws_temp_interp"].to_numpy()
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    print(f"XGBoost  RMSE {rmse(aws + pred - obs):.3f} °C  MAE {np.mean(np.abs(aws + pred - obs)):.3f} °C")
    print(f"AWS base RMSE {rmse(aws - obs):.3f} °C  MAE {np.mean(np.abs(aws - obs)):.3f} °C")
    print(f"xgboost {xgb.__version__}, {ROUNDS} rounds, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
