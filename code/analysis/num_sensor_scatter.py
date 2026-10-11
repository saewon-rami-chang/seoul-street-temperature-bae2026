"""Observed vs predicted sensor-mean residuals at LORO holdout sensors (P1)."""

import json

import numpy as np
import pandas as pd
import polars as pl

from numlib import HERE, TRAIN, REGION_LABEL, save, scan_cache

RUNS = {"Reference_GPU": TRAIN / "Reference_GPU", "Ablation_spatial": TRAIN / "Ablation_spatial"}


def sensor_means(d):
    p = pl.read_parquet(d / "Predictions/Predictions.parquet",
                        columns=["serial_no", "region", "target_resid", "pred_resid"])
    return (p.group_by("serial_no").agg(pl.col("region").first(), pl.len().alias("N_hours"),
                                        pl.col("target_resid").mean().alias("obs"),
                                        pl.col("pred_resid").mean().alias("pred")).to_pandas().set_index("serial_no"))


def main():
    a, b = (sensor_means(d) for d in RUNS.values())
    assert set(a.index) == set(b.index)
    assert np.allclose(a.obs, b.obs.reindex(a.index)), "observed residuals differ between runs"
    sm = pd.read_csv(TRAIN / "Discovery_Outputs/Spatial/Sensor_Map.csv").set_index("serial_no")
    svf = (scan_cache().select(["serial_no", "SVF_10m"]).group_by("serial_no")
           .agg(pl.col("SVF_10m").drop_nulls().first()).collect().to_pandas().set_index("serial_no")["SVF_10m"])
    df = pd.DataFrame({
        "serial_no": a.index,
        "generation": np.where(a.index.str.startswith("OC3"), "OC3", "V02Q"),
        "region": a.region.map(REGION_LABEL).values,
        "N_hours": a.N_hours.values,
        "obs_mean_resid": a.obs.values,
        "pred_mean_resid_Reference_GPU": a.pred.values,
        "pred_mean_resid_Ablation_spatial": b.pred.reindex(a.index).values,
    })
    df["error_Reference_GPU"] = df.pred_mean_resid_Reference_GPU - df.obs_mean_resid
    df["error_Ablation_spatial"] = df.pred_mean_resid_Ablation_spatial - df.obs_mean_resid
    df = df.join(sm[["lat", "lon", "x", "y"]], on="serial_no").join(svf.rename("SVF_10m"), on="serial_no")
    df = df.sort_values(["region", "generation", "serial_no"]).reset_index(drop=True)
    save(df, "P1_sensor_means_scatter.csv", "%.5f")
    for g, s in [("all", df), ("V02Q", df[df.generation == "V02Q"]), ("OC3", df[df.generation == "OC3"])]:
        r = {k: np.corrcoef(s.obs_mean_resid, s[f"pred_mean_resid_{k}"])[0, 1] for k in RUNS}
        print(f"{g:5s} n={len(s):3d}  r(obs, pred) Reference_GPU {r['Reference_GPU']:.3f}  "
              f"Ablation_spatial {r['Ablation_spatial']:.3f}")


if __name__ == "__main__":
    main()
