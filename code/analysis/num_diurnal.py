"""Hourly errors per run (D1)."""

import pandas as pd
import polars as pl

from numlib import HERE, TRAIN, save

RUNS = {"Main (CPU, published)": TRAIN / "Discovery_Outputs"}
RUNS.update({d.name: d for d in sorted(TRAIN.iterdir())
             if (d / "Predictions/Predictions.parquet").exists()
             and d.name not in ("Discovery_Outputs", "Random8020_Outputs")})


def main():
    out = []
    for run, d in RUNS.items():
        p = (pl.read_parquet(d / "Predictions/Predictions.parquet", columns=["datetime", "actual_temp", "pred_temp"])
             .with_columns(pl.col("datetime").cast(pl.Datetime).dt.hour().alias("Hour"),
                           (pl.col("pred_temp").cast(pl.Float64) - pl.col("actual_temp").cast(pl.Float64)).alias("e")))
        h = (p.group_by("Hour").agg(pl.len().alias("N_obs"), pl.col("e").abs().mean().alias("MAE"),
                                    pl.col("e").abs().median().alias("Median_AE"),
                                    (pl.col("e") ** 2).mean().sqrt().alias("RMSE"), pl.col("e").mean().alias("MBE"))
             .sort("Hour").to_pandas())
        ds = pd.read_csv(d / "Metrics/Diurnal_Stats.csv").add_prefix("ErrDist_").rename(columns={"ErrDist_Hour": "Hour"})
        h = h.merge(ds, on="Hour", how="left")
        h.insert(0, "Run", run)
        out.append(h)
        print(f"{run:24s} MAE min {h.MAE.min():.3f} at {int(h.loc[h.MAE.idxmin(), 'Hour']):02d}h, "
              f"max {h.MAE.max():.3f} at {int(h.loc[h.MAE.idxmax(), 'Hour']):02d}h")
    save(pd.concat(out, ignore_index=True), "D1_diurnal_by_run.csv")


if __name__ == "__main__":
    main()
