"""Every run with CIs, and paired differences between runs (A1, A2)."""

from pathlib import Path

import zlib

import numpy as np
import pandas as pd
import polars as pl

from numlib import NUMBERS, TRAIN

HERE = Path(__file__).resolve().parent
RUNS = {  # label -> output folder (results/<run>/ written by code/train_loro.py)
    "Main (CPU, published)": TRAIN / "Discovery_Outputs",
    "Reference_GPU":         TRAIN / "Reference_GPU",
    "Ablation_spatial":      TRAIN / "Ablation_spatial",
    "Ablation_shadow":       TRAIN / "Ablation_shadow",
    "Ablation_absolute":     TRAIN / "Ablation_absolute",
    "Holdout_ST":            TRAIN / "Holdout_ST",
    "Ablation_nearest":      TRAIN / "Ablation_nearest",
    "Ablation_shadow_svf":   TRAIN / "Ablation_shadow_svf",
    "Ablation_v02q_only":    TRAIN / "Ablation_v02q_only",
    "Ablation_scale10":      TRAIN / "Ablation_scale10",
    "Ablation_scale100":     TRAIN / "Ablation_scale100",
    "Ablation_scale250":     TRAIN / "Ablation_scale250",
    "Ablation_scale1000":    TRAIN / "Ablation_scale1000",
}
REGION_ORDER = ["A: Mapo / Sangam DMC", "B: Gangnam / Seocho CBD", "C: Central Seoul",
                "D: Gangdong (East)", "E: Dobong / Nowon (North)", "F: Gwanak / Sillim", "Pooled"]
N_BOOT = 1000


def rng_for(label: str):
    """Independent, fixed random stream per run / comparison, so adding runs never changes existing CIs."""
    return np.random.default_rng([SEED, zlib.crc32(label.encode())])
SEED = 42


def sensor_sums(pred: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-sensor sufficient statistics of the error pred_col - actual_temp."""
    err = pred[col].to_numpy(np.float64) - pred["actual_temp"].to_numpy(np.float64)
    df = pd.DataFrame({"serial_no": pred["serial_no"].to_numpy(), "region": pred["region"].to_numpy(),
                       "n": 1.0, "e": err, "e2": err ** 2, "ae": np.abs(err)})
    return df.groupby(["region", "serial_no"], sort=False)[["n", "e", "e2", "ae"]].sum().reset_index()


def bootstrap_ci(sums: pd.DataFrame, rng) -> dict:
    """95 % CI of RMSE/MAE/MBE per region and pooled; sensors resampled within region."""
    out = {}
    pooled = np.zeros((N_BOOT, 4))
    for region, g in sums.groupby("region", sort=False):
        s = g[["n", "e", "e2", "ae"]].to_numpy()
        w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=N_BOOT)  # (N_BOOT, n_sensors)
        tot = w @ s                                                              # (N_BOOT, 4)
        pooled += tot
        out[region] = tot
    out["Pooled"] = pooled
    ci = {}
    for region, tot in out.items():
        n = tot[:, 0]
        stats = {"RMSE": np.sqrt(tot[:, 2] / n), "MAE": tot[:, 3] / n, "MBE": tot[:, 1] / n}
        ci[region] = {f"{k}_{b}": float(np.percentile(v, q))
                      for k, v in stats.items() for b, q in (("lo", 2.5), ("hi", 97.5))}
    return ci


PAIRS = [  # (label, run_a, col_a, run_b, col_b): CI of metric(b) - metric(a) on the same sensors
    ("GPU reproduction: Reference_GPU - Main", "Main (CPU, published)", "pred_temp", "Reference_GPU", "pred_temp"),
    ("No spatial features - Reference",        "Reference_GPU", "pred_temp", "Ablation_spatial", "pred_temp"),
    ("No shadow - Reference",                  "Reference_GPU", "pred_temp", "Ablation_shadow", "pred_temp"),
    ("Absolute target - Reference",            "Reference_GPU", "pred_temp", "Ablation_absolute", "pred_temp"),
    ("Reference - AWS baseline",               "Reference_GPU", "aws_temp_interp", "Reference_GPU", "pred_temp"),
    ("Holdout_ST - its AWS baseline",          "Holdout_ST", "aws_temp_interp", "Holdout_ST", "pred_temp"),
    ("Nearest-neighbour features - Reference (bilinear)", "Reference_GPU", "pred_temp", "Ablation_nearest", "pred_temp"),
    ("Neither SVF nor shadow - Reference",     "Reference_GPU", "pred_temp", "Ablation_shadow_svf", "pred_temp"),
    ("Neither SVF nor shadow - No shadow (SVF only)", "Ablation_shadow", "pred_temp", "Ablation_shadow_svf", "pred_temp"),
    # restrict run a to run b's sensors (V02Q-only run)
    ("V02Q-only training - Reference (257 V02Q holdout sensors)", "Reference_GPU", "pred_temp",
     "Ablation_v02q_only", "pred_temp", "restrict"),
] + [p for s in ("10", "100", "250", "1000") for p in (
    (f"Single scale {s} m - Reference", "Reference_GPU", "pred_temp", f"Ablation_scale{s}", "pred_temp"),
    (f"Single scale {s} m - No spatial features", "Ablation_spatial", "pred_temp", f"Ablation_scale{s}", "pred_temp"))]


def paired_diff(sa: pd.DataFrame, sb: pd.DataFrame, rng, restrict=False) -> list[dict]:
    """Bootstrap CI of metric(b) - metric(a); same sensor resample for a and b.
    restrict=True: a is reduced to b's sensors (b must be a subset of a)."""
    m = sa.merge(sb, on=["region", "serial_no"], suffixes=("_a", "_b"))
    if len(m) < len(sb) or (not restrict and len(m) < len(sa)):
        raise ValueError("paired runs do not share the same sensors")
    cols = ["n", "e", "e2", "ae"]
    acc = {}
    pooled_a = np.zeros((N_BOOT, 4)); pooled_b = np.zeros((N_BOOT, 4))
    for region, g in m.groupby("region", sort=False):
        w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=N_BOOT)
        ta = w @ g[[c + "_a" for c in cols]].to_numpy(); tb = w @ g[[c + "_b" for c in cols]].to_numpy()
        pooled_a += ta; pooled_b += tb
        acc[region] = (ta, tb)
    acc["Pooled"] = (pooled_a, pooled_b)
    point = {"a": m[[c + "_a" for c in cols]], "b": m[[c + "_b" for c in cols]]}
    rows = []
    for region in REGION_ORDER:
        if region not in acc:
            continue
        ta, tb = acc[region]
        sel = m if region == "Pooled" else m[m["region"] == region]
        pa = sel[[c + "_a" for c in cols]].sum().to_numpy(); pb = sel[[c + "_b" for c in cols]].sum().to_numpy()
        for name, f in (("RMSE", lambda t: np.sqrt(t[..., 2] / t[..., 0])),
                        ("MAE", lambda t: t[..., 3] / t[..., 0]),
                        ("MBE", lambda t: t[..., 1] / t[..., 0])):
            d = f(tb) - f(ta)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"Region": region, "Metric": name, "Diff": float(f(pb) - f(pa)),
                         "CI_lo": float(lo), "CI_hi": float(hi),
                         "Excludes_0": bool(lo > 0 or hi < 0), "N_sensors": len(sel)})
    return rows


def region_label(key: str) -> str:
    return next((r for r in REGION_ORDER if r.startswith(key.split("_")[0] + ":")), key)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    sums = {}  # (run, column) -> per-sensor sufficient statistics, reused for paired CIs
    for run, d in RUNS.items():
        d = d.resolve()
        if not (d / "Metrics/Global_Metrics_Absolute.csv").exists():
            print(f"skip {run}: no metrics in {d}")
            continue
        ab = pd.read_csv(d / "Metrics/Global_Metrics_Absolute.csv")
        rs = pd.read_csv(d / "Metrics/Global_Metrics_Residual.csv").set_index("Region")
        pred_path = d / "Predictions/Predictions.parquet"
        ci_model = ci_base = {}
        if pred_path.exists():
            # read with polars: pyarrow 19 fails on these files
            pred = pl.read_parquet(pred_path, columns=["serial_no", "region", "actual_temp",
                                                       "pred_temp", "aws_temp_interp"]).to_pandas()
            pred["region"] = pred["region"].map(region_label)
            for col in ("pred_temp", "aws_temp_interp"):
                sums[(run, col)] = sensor_sums(pred, col)
            del pred
            ci_model = bootstrap_ci(sums[(run, "pred_temp")], rng_for(run + "|model"))
            ci_base = bootstrap_ci(sums[(run, "aws_temp_interp")], rng_for(run + "|baseline"))
        for _, r in ab.iterrows():
            model = "XGBoost" if r["Model"] == "XGBoost" else "AWS baseline"
            ci = (ci_model if model == "XGBoost" else ci_base).get(r["Region"], {})
            rows.append({
                "Run": run, "Model": model, "Region": r["Region"], "N_sensors": r["N_sensors"],
                "N_obs": r["N"], "RMSE": r["RMSE"], "MAE": r["MAE"], "MBE": r["MBE"],
                "R2_abs": r["R2"],
                "R2_resid": rs.loc[r["Region"], "R2"] if model == "XGBoost" else np.nan,
                **ci,
            })
    df = pd.DataFrame(rows)
    df["Region"] = pd.Categorical(df["Region"], REGION_ORDER, ordered=True)
    df = df.sort_values(["Model", "Region", "Run"], key=lambda s: s.map(list(RUNS).index)
                        if s.name == "Run" else s).reset_index(drop=True)
    out = NUMBERS
    out.mkdir(exist_ok=True)
    df.to_csv(out / "A1_run_comparison.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: "" if pd.isna(v) else f"{v:.3f}"
    show = df[df["Model"] == "XGBoost"].copy()
    for m in ("RMSE", "MAE", "MBE"):
        if f"{m}_lo" in show:
            show[f"{m} 95% CI"] = [f"[{fmt(a)}, {fmt(b)}]" if pd.notna(a) else ""
                                   for a, b in zip(show[f"{m}_lo"], show[f"{m}_hi"])]
    cols = ["Run", "Region", "N_sensors", "RMSE", "MAE", "MBE", "R2_resid"] + \
           [c for c in show.columns if c.endswith("95% CI")]
    print(show[show["Region"] == "Pooled"][cols].to_string(index=False, float_format="%.3f"))
    print()
    wide = show.pivot_table(index="Region", columns="Run", values="MAE", observed=True)[list(
        r for r in RUNS if r in show["Run"].unique())]
    print("MAE by region (°C):")
    print(wide.to_string(float_format="%.3f"))
    base = df[(df["Model"] == "AWS baseline") & (df["Region"] == "Pooled")]
    print("\nAWS baseline, pooled:")
    print(base[["Run", "N_sensors", "RMSE", "MAE", "MBE"]].to_string(index=False, float_format="%.3f"))
    print(f"\nSaved {out / "A1_run_comparison.csv"}")

    prow = []
    for label, ra, ca, rb, cb, *opt in PAIRS:
        if (ra, ca) in sums and (rb, cb) in sums:
            prow += [{"Comparison": label, **r}
                     for r in paired_diff(sums[(ra, ca)], sums[(rb, cb)], rng_for(label), restrict="restrict" in opt)]
    if prow:
        pdf = pd.DataFrame(prow)
        pdf.to_csv(out / "A2_paired_differences.csv", index=False)
        pooled = pdf[pdf["Region"] == "Pooled"].copy()
        pooled["95% CI"] = [f"[{a:+.3f}, {b:+.3f}]" for a, b in zip(pooled["CI_lo"], pooled["CI_hi"])]
        print("\nPaired differences, pooled (b - a, °C; positive RMSE/MAE = b is worse):")
        print(pooled[["Comparison", "Metric", "Diff", "95% CI", "Excludes_0"]]
              .to_string(index=False, float_format="%+.3f"))
        print(f"Saved {out / 'A2_paired_differences.csv'}")


if __name__ == "__main__":
    main()
