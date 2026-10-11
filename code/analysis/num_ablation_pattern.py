"""Between-sensor variance explained per run (A3, A4, H6). Run num_validation.py and num_hardware.py first."""

import numpy as np
import pandas as pd
import polars as pl

from numlib import HERE, NUMBERS, TRAIN, save, scan_cache

RUNS = {"Main (CPU, published)": TRAIN / "Discovery_Outputs", "Reference_GPU": TRAIN / "Reference_GPU",
        "Ablation_spatial": TRAIN / "Ablation_spatial", "Ablation_shadow": TRAIN / "Ablation_shadow",
        "Ablation_absolute": TRAIN / "Ablation_absolute",
        "Ablation_nearest": TRAIN / "Ablation_nearest", "Ablation_shadow_svf": TRAIN / "Ablation_shadow_svf",
        **{f"Ablation_scale{s}": TRAIN / f"Ablation_scale{s}" for s in ("10", "100", "250", "1000")}}
RUNS = {k: d for k, d in RUNS.items() if (d / "Predictions/Predictions.parquet").exists()}  # skip runs not yet back
# also compare single-scale runs with the no-spatial run
EXTRA_REF = {f"Ablation_scale{s}": "Ablation_spatial" for s in ("10", "100", "250", "1000")}
# V02Q-only runs: compared in the V02Q variants only
RUNS_V02Q = {k: d for k, d in {"Ablation_v02q_only": TRAIN / "Ablation_v02q_only"}.items()
             if (d / "Predictions/Predictions.parquet").exists()}
VARIANTS = {"all": "", "v02q": "_v02q", "oc3": "_oc3", "genadj": "_genadj", "v02q_open": "_v02q_open"}
N_BOOT = 1000


def sensor_means(d):
    p = pl.read_parquet(d / "Predictions/Predictions.parquet", columns=["serial_no", "target_resid", "pred_resid"])
    return (p.group_by("serial_no").agg(pl.col("target_resid").mean().alias("obs"),
                                        pl.col("pred_resid").mean().alias("pred")).to_pandas()
            .set_index("serial_no"))


def share(err, obs):
    return 1 - np.var(err, ddof=1) / np.var(obs, ddof=1)


def gen_demean(obs, oc3):
    """Subtract each generation's mean observed residual."""
    out = obs.copy()
    for g in (True, False):
        out[oc3 == g] -= obs[oc3 == g].mean()
    return out


def run_variant(variant, means, ids, rng, svf=None):
    oc3_all = np.array([s.startswith("OC3") for s in ids])
    thr = np.nanmedian(svf[oc3_all]) if svf is not None else np.nan
    keep = {"v02q": ~oc3_all, "oc3": oc3_all,
            "v02q_open": (~oc3_all & (svf > thr)) if svf is not None else None}.get(variant)
    keep = np.ones(len(ids), bool) if keep is None else keep
    sel = [s for s, k in zip(ids, keep) if k]
    oc3 = oc3_all[keep]
    obs_raw = means["Reference_GPU"].loc[sel, "obs"].to_numpy()
    pred = {k: m.loc[sel, "pred"].to_numpy() for k, m in means.items()}

    def stats_for(idx):
        o = obs_raw[idx]
        if variant == "genadj":
            o = gen_demean(o, oc3[idx])
        return o, {k: share(p[idx] - o, o) for k, p in pred.items()}

    o_full, point = stats_for(np.arange(len(sel)))
    boot_idx = rng.integers(0, len(sel), (N_BOOT, len(sel)))
    bs = {k: np.empty(N_BOOT) for k in pred}
    for b, idx in enumerate(boot_idx):
        _, s = stats_for(idx)
        for k in pred:
            bs[k][b] = s[k]
    rows = [{"Variant": variant, "Run": k, "N_sensors": len(sel), "N_OC3": int(oc3.sum()),
             "SVF10m_threshold_OC3_median": thr if variant == "v02q_open" else np.nan,
             "Median_SVF10m_of_sensors_used": float(np.nanmedian(svf[keep])) if svf is not None else np.nan,
             "SD_obs_sensor_mean_resid": o_full.std(ddof=1),
             "SD_sensor_mean_error": (pred[k] - o_full).std(ddof=1), "Share_explained": point[k],
             "CI_lo": np.percentile(bs[k], 2.5), "CI_hi": np.percentile(bs[k], 97.5)} for k in pred]
    prow = []
    pairs = [(k, "Reference_GPU") for k in pred if k != "Reference_GPU"]
    pairs += [(k, r) for k, r in EXTRA_REF.items() if k in pred and r in pred]
    for k, ref in pairs:
        d = bs[k] - bs[ref]
        prow.append({"Variant": variant, "Comparison": f"{k} - {ref}",
                     "Diff_share_explained": point[k] - point[ref],
                     "CI_lo": np.percentile(d, 2.5), "CI_hi": np.percentile(d, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(prow)


def main():
    means = {k: sensor_means(d) for k, d in RUNS.items()}
    means_v02q = {k: sensor_means(d) for k, d in RUNS_V02Q.items()}
    ids = sorted(set.intersection(*(set(m.index) for m in means.values())))
    svf_s = (scan_cache().select(["serial_no", "SVF_10m"]).group_by("serial_no")
             .agg(pl.col("SVF_10m").drop_nulls().first()).collect().to_pandas().set_index("serial_no")["SVF_10m"])
    svf = svf_s.reindex(ids).to_numpy()
    pd.set_option("display.width", 200)
    for variant, suffix in VARIANTS.items():
        rng = np.random.default_rng(42)  # same resamples as the published 'all' variant
        m = {**means, **means_v02q} if variant in ("v02q", "v02q_open") else means
        a3, a4 = run_variant(variant, m, ids, rng, svf)
        save(a3, f"A3{suffix}_between_sensor_pattern.csv")
        save(a4, f"A4{suffix}_between_sensor_paired.csv")
        print(a3.to_string(index=False, float_format="%.3f"))
        print(a4.to_string(index=False, float_format="%.3f"), "\n")

    h1 = pd.read_csv(NUMBERS / "H1_generation_by_region.csv")
    v1 = pd.read_csv(NUMBERS / "V1_validation_tables_CI.csv")
    t6 = v1[(v1.Scheme == "Spatial LORO") & (v1.Model == "XGBoost") & (v1.Subset != "All")][["Subset", "MBE", "RMSE"]]
    ho = h1[h1.Zone == "Holdout"][["Region", "OC3_share", "N"]]
    h6 = t6.merge(ho, left_on="Subset", right_on="Region").drop(columns="Subset")
    r = np.corrcoef(h6.OC3_share, h6.MBE)[0, 1]
    h6 = pd.concat([h6, pd.DataFrame([{"Region": f"Pearson r (OC3 share vs MBE, 6 regions) = {r:.2f}"}])])
    save(h6, "H6_region_bias_vs_OC3_share.csv")


if __name__ == "__main__":
    main()
