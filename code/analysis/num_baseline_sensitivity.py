"""AWS-baseline sensitivity to k, p and lapse rate (B1). Needs extra_inputs/."""

import json

import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer
from scipy import stats

from numlib import SEOUL, TRAIN, REGION_LABEL, save, scan_cache

PRE = SEOUL / "01_preprocess"
CACHE = SEOUL / "02_cache/chunks"
DEFAULT = {"k": 15, "p": 2, "gamma": -6.5}
SETTINGS = ([{**DEFAULT, "k": k} for k in (10, 15, 20)] + [{**DEFAULT, "p": p} for p in (1, 3)]
            + [{**DEFAULT, "gamma": 0.0}])
MIN_VALID = 3


def holdout_sensors():
    sm = pd.read_csv(TRAIN / "Discovery_Outputs/Spatial/Sensor_Map.csv")
    bounds = json.load(open(TRAIN / "Discovery_Outputs/Spatial/Region_Bounds_5179.json"))
    keep = np.zeros(len(sm), bool)
    for b in bounds.values():
        keep |= sm.x.between(b["x_min"], b["x_max"]).to_numpy() & sm.y.between(b["y_min"], b["y_max"]).to_numpy()
    return sorted(sm.loc[keep, "serial_no"])


def main():
    sids = holdout_sensors()
    print(f"holdout sensors: {len(sids)}")

    # sensor coordinates exactly as the builder (locations file, projected)
    loc = pd.read_csv(PRE / "02_sdot/sdot_station_locations.csv").dropna(subset=["lat", "lon"])
    loc = loc.set_index("serial_no").loc[sids]
    sx, sy = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True).transform(loc.lon.values, loc.lat.values)

    aws = pl.read_parquet(PRE / "01_aws/aws_clean.parquet",
                          columns=["station_id", "datetime", "temperature", "x_coord", "y_coord", "altitude"]).to_pandas()
    aws["datetime"] = pd.to_datetime(aws["datetime"])
    meta = aws.groupby("station_id").agg(x=("x_coord", "first"), y=("y_coord", "first"),
                                         alt=("altitude", "first")).reset_index()

    # cache rows for these sensors: observed temperature, published baseline, DEM_10m
    rows = (scan_cache()
            .filter(pl.col("serial_no").is_in(sids) & pl.col("target_resid").is_not_null()
                    & (pl.col("target_resid").abs() < 25))
            .select(["serial_no", "datetime", "aws_temp_interp", "target_resid", "DEM_10m"]).collect().to_pandas())
    rows["datetime"] = pd.to_datetime(rows["datetime"])
    rows["obs"] = rows.aws_temp_interp.astype(np.float64) + rows.target_resid.astype(np.float64)
    dem = rows.groupby("serial_no").DEM_10m.first().reindex(sids)
    sensor_alt = np.where(dem.isna(), 0.0, dem.to_numpy(np.float64))
    print(f"rows: {len(rows):,}")

    hours = pd.DatetimeIndex(sorted(rows.datetime.unique()))
    piv = (aws[aws.datetime.isin(hours)].pivot_table(index="datetime", columns="station_id", values="temperature",
                                                     aggfunc="first").reindex(index=hours, columns=meta.station_id))
    T = piv.to_numpy(np.float64)                                   # (n_hours, n_stations)
    d_all = np.hypot(sx[:, None] - meta.x.to_numpy()[None], sy[:, None] - meta.y.to_numpy()[None])
    s_idx = pd.Series(np.arange(len(sids)), index=sids)
    r_s = s_idx.loc[rows.serial_no].to_numpy()
    r_h = hours.get_indexer(rows.datetime)

    def baseline(k, p, gamma):
        order = np.argsort(d_all, axis=1)[:, :k]                   # k nearest, sorted
        dist = np.take_along_axis(d_all, order, axis=1)
        w = 1.0 / dist ** p
        lapse = (gamma / 1000.0) * (sensor_alt[:, None] - meta.alt.to_numpy()[order])
        out = np.empty(len(rows))
        for s in range(len(sids)):                                 # all hours at once per sensor
            m = r_s == s
            g = T[r_h[m]][:, order[s]] + lapse[s]                  # (n_rows_s, k)
            valid = ~np.isnan(g)
            ww = np.where(valid, w[s], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (ww * np.where(valid, g, 0.0)).sum(1) / ww.sum(1)
            v[valid.sum(1) < MIN_VALID] = np.nan
            out[m] = v
        return out

    base_def = baseline(**DEFAULT)
    diff = np.abs(base_def - rows.aws_temp_interp.to_numpy(np.float64))
    check = {"max_abs_diff_vs_cache": float(np.nanmax(diff)), "median_abs_diff": float(np.nanmedian(diff)),
             "rows_nan": int(np.isnan(base_def).sum())}
    print("default vs cache baseline:", check)

    def sensor_means(b):
        r = pd.DataFrame({"s": rows.serial_no.to_numpy(), "res": rows.obs.to_numpy() - b})
        return r.groupby("s").res.mean().reindex(sids)

    ref = sensor_means(base_def)
    out = []
    for st in SETTINGS:
        b = base_def if st == DEFAULT else baseline(**st)
        mres = sensor_means(b)
        ok = ref.notna() & mres.notna()
        changed = [k for k in DEFAULT if st[k] != DEFAULT[k]]
        out.append({"Setting": "default" if not changed else f"{changed[0]} = {st[changed[0]]:g}",
                    "k": st["k"], "p": st["p"], "Gamma_C_per_km": st["gamma"], "N_sensors": int(ok.sum()),
                    "Pearson_r_vs_default": stats.pearsonr(ref[ok], mres[ok])[0],
                    "Spearman_rho_vs_default": stats.spearmanr(ref[ok], mres[ok])[0],
                    "SD_sensor_means": mres[ok].std(ddof=1), "Mean_sensor_means": mres[ok].mean(),
                    "Mean_abs_change_vs_default": (mres[ok] - ref[ok]).abs().mean(),
                    "Rows_with_baseline": int(np.isfinite(b).sum())})
        print(out[-1])
    df = pd.DataFrame(out)
    df = pd.concat([df, pd.DataFrame([{"Setting": f"CHECK default reproduces cache baseline: max |diff| "
                                                  f"{check['max_abs_diff_vs_cache']:.2e} °C, NaN rows {check['rows_nan']}"}])])
    save(df, "B1_baseline_sensitivity.csv")
    print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
