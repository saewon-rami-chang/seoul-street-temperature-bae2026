"""V02Q vs OC3 hardware generations (H1-H5)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from scipy.spatial import cKDTree

from numlib import NUMBERS, PRED, REGION_LABEL, SEOUL, TRAIN, load_pred, metrics, save, scan_cache, sensor_ci

CACHE = SEOUL / "02_cache/chunks"
BUFFER_M = 1000
N_TOP = 15
rng = np.random.default_rng(42)


def generation(s: pd.Series) -> pd.Series:
    return np.where(s.str.startswith("V02Q"), "V02Q", np.where(s.str.startswith("OC3"), "OC3", "other"))


def sensors_with_regions() -> pd.DataFrame:
    sm = pd.read_csv(TRAIN / "Discovery_Outputs/Spatial/Sensor_Map.csv")
    bounds = json.load(open(TRAIN / "Discovery_Outputs/Spatial/Region_Bounds_5179.json"))
    sm["generation"] = generation(sm["serial_no"])
    sm["zone"], sm["region"] = "Outside regions", "-"
    for key, b in bounds.items():
        inb = sm.x.between(b["x_min"], b["x_max"]) & sm.y.between(b["y_min"], b["y_max"])
        buf = (sm.x.between(b["x_min"] - BUFFER_M, b["x_max"] + BUFFER_M)
               & sm.y.between(b["y_min"] - BUFFER_M, b["y_max"] + BUFFER_M) & ~inb)
        sm.loc[inb, ["zone", "region"]] = ["Holdout", REGION_LABEL[key]]
        sm.loc[buf & (sm.zone == "Outside regions"), ["zone", "region"]] = ["Buffer", REGION_LABEL[key]]
    return sm, bounds


def top_spatial_features() -> list[str]:
    d = pd.read_csv(TRAIN / "Full_Model/SHAP/SHAP_Full.csv")
    sp = d[d.Feature.str.match(r".*_(?:10|100|250|1000)m$") & ~d.Feature.str.startswith("shadow_fraction")]
    return sp.nlargest(N_TOP, "Mean_Abs_SHAP")["Feature"].tolist()


def static_features(feats) -> pd.DataFrame:
    return (scan_cache().select(["serial_no", *feats])
            .group_by("serial_no").agg([pl.col(f).drop_nulls().first() for f in feats])
            .collect().to_pandas())


def h1(sm):
    rows = []
    for (zone, region), g in sm.groupby(["zone", "region"]):
        rows.append({"Zone": zone, "Region": region, "N": len(g), "V02Q": int((g.generation == "V02Q").sum()),
                     "OC3": int((g.generation == "OC3").sum()), "OC3_share": (g.generation == "OC3").mean()})
    rows.append({"Zone": "All sensors", "Region": "Seoul", "N": len(sm), "V02Q": int((sm.generation == "V02Q").sum()),
                 "OC3": int((sm.generation == "OC3").sum()), "OC3_share": (sm.generation == "OC3").mean()})
    df = pd.DataFrame(rows)
    ho = sm[sm.zone == "Holdout"]
    ct = pd.crosstab(ho.region, ho.generation)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    ct2 = pd.crosstab(sm.zone == "Holdout", sm.generation)
    chi2b, pb, _, _ = stats.chi2_contingency(ct2)
    df = pd.concat([df, pd.DataFrame([
        {"Zone": "TEST", "Region": f"chi2 holdout region x generation: chi2={chi2:.2f}, dof={dof}, p={p:.3g}"},
        {"Zone": "TEST", "Region": f"chi2 in-any-holdout x generation: chi2={chi2b:.2f}, p={pb:.3g}"}])])
    return df


def h2(sm, k=5, n_perm=2000):
    xy = sm[["x", "y"]].to_numpy()
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    gen = (sm.generation == "OC3").to_numpy()
    nb = idx[:, 1:]
    obs = {g: (gen[nb] == gen[:, None])[gen == (g == "OC3")].mean() for g in ("OC3", "V02Q")}
    null = {g: [] for g in obs}
    for _ in range(n_perm):
        pg = rng.permutation(gen)
        same = pg[nb] == pg[:, None]
        for g in obs:
            null[g].append(same[pg == (g == "OC3")].mean())
    dist = cKDTree(xy[gen]).query(xy[gen], k=2)[0][:, 1]
    dist_v = cKDTree(xy[~gen]).query(xy[gen], k=1)[0]
    rows = [{"Generation": g, "k": k, "Observed_same_gen_share": obs[g],
             "Null_mean": np.mean(null[g]), "Null_95hi": np.percentile(null[g], 97.5),
             "Perm_p_one_sided": (np.sum(np.array(null[g]) >= obs[g]) + 1) / (n_perm + 1)} for g in obs]
    rows.append({"Generation": "OC3 median NN distance to OC3 (m)", "Observed_same_gen_share": np.median(dist)})
    rows.append({"Generation": "OC3 median NN distance to V02Q (m)", "Observed_same_gen_share": np.median(dist_v)})
    return pd.DataFrame(rows)


def h3(feat_df, feats):
    rows = []
    for f in feats:
        a = feat_df.loc[feat_df.generation == "OC3", f].dropna(); b = feat_df.loc[feat_df.generation == "V02Q", f].dropna()
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Feature": f, "Mean_OC3": a.mean(), "Mean_V02Q": b.mean(), "Median_OC3": a.median(),
                     "Median_V02Q": b.median(), "SMD_OC3_minus_V02Q": (a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                     "MannWhitney_p": stats.mannwhitneyu(a, b).pvalue, "N_OC3": len(a), "N_V02Q": len(b)})
    return pd.DataFrame(rows)


def ols_coef(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def h4_h5(sm, feat_df, feats):
    h4, h5 = [], []
    gen_of = dict(zip(sm.serial_no, sm.generation))
    for scheme in ("Spatial LORO", "Temporal 2025"):
        df = load_pred(PRED[scheme])
        df["generation"] = df["serial_no"].map(gen_of).fillna(pd.Series(generation(df["serial_no"]), index=df.index))
        for g in ("V02Q", "OC3"):
            sub = df[df.generation == g]
            h4.append({"Scheme": scheme, "Generation": g, **metrics(sub.actual_temp, sub.pred_temp),
                       "R2_residual": metrics(sub.target_resid, sub.pred_resid)["R2"], **sensor_ci(sub, rng=rng)})
        ps = df.groupby("serial_no").agg(obs_resid=("target_resid", "mean"), pred_resid=("pred_resid", "mean"),
                                         n=("target_resid", "size")).reset_index()
        ps["error"] = ps.pred_resid - ps.obs_resid
        ps["generation"] = ps.serial_no.map(gen_of).fillna(pd.Series(generation(ps.serial_no), index=ps.index))
        ps = ps.merge(feat_df[["serial_no", *feats]], on="serial_no", how="left")
        Z = ps[feats].apply(lambda c: (c - c.mean()) / c.std()).fillna(0).to_numpy()
        oc3 = (ps.generation == "OC3").to_numpy().astype(float)
        X_raw = np.column_stack([np.ones(len(ps)), oc3])
        X_adj = np.column_stack([X_raw, Z])
        for var in ("obs_resid", "pred_resid", "error"):
            y = ps[var].to_numpy()
            est = {"raw": ols_coef(X_raw, y)[1], "adj": ols_coef(X_adj, y)[1]}
            boot = {"raw": [], "adj": []}
            for _ in range(1000):
                i = rng.integers(0, len(ps), len(ps))
                boot["raw"].append(ols_coef(X_raw[i], y[i])[1]); boot["adj"].append(ols_coef(X_adj[i], y[i])[1])
            for kind in ("raw", "adj"):
                lo, hi = np.percentile(boot[kind], [2.5, 97.5])
                h5.append({"Scheme": scheme, "Quantity": f"sensor time-mean {var}",
                           "Comparison": "OC3 - V02Q" + (" (adjusted for top-15 spatial features)" if kind == "adj" else ""),
                           "Difference_C": est[kind], "CI_lo": lo, "CI_hi": hi,
                           "Mean_V02Q": ps.loc[oc3 == 0, var].mean(), "Mean_OC3": ps.loc[oc3 == 1, var].mean(),
                           "N_V02Q": int((oc3 == 0).sum()), "N_OC3": int(oc3.sum())})
        del df
    return pd.DataFrame(h4), pd.DataFrame(h5)


def figure(sm, bounds):
    import geopandas as gpd
    out = NUMBERS / "figures"; out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        gpd.read_file(SEOUL / "03_training_new/seoul_boundary.geojson").to_crs(5179).boundary.plot(
            ax=ax, color="0.4", lw=0.6)
    except Exception as e:
        print("  boundary not drawn:", e)
    for g, c, m in (("V02Q", "#4C72B0", "o"), ("OC3", "#DD8452", "^")):
        s = sm[sm.generation == g]
        ax.scatter(s.x, s.y, s=7 if g == "V02Q" else 14, c=c, marker=m, lw=0, label=f"{g} (n={len(s)})", zorder=3)
    for key, b in bounds.items():
        ax.add_patch(plt.Rectangle((b["x_min"], b["y_min"]), b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
                                   fill=False, ec="k", lw=1.2, zorder=4))
        ax.add_patch(plt.Rectangle((b["x_min"] - BUFFER_M, b["y_min"] - BUFFER_M), b["x_max"] - b["x_min"] + 2 * BUFFER_M,
                                   b["y_max"] - b["y_min"] + 2 * BUFFER_M, fill=False, ec="k", lw=0.6, ls="--", zorder=4))
        ax.text(b["x_min"], b["y_max"] + 150, REGION_LABEL[key], fontsize=7, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("S-DoT hardware generation and LORO regions (solid: holdout, dashed: 1 km buffer)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "H_generation_map.png", dpi=250); plt.close(fig)
    print(f"  -> numbers/figures/H_generation_map.png")


def main():
    sm, bounds = sensors_with_regions()
    feats = top_spatial_features()
    print("top spatial features:", feats)
    feat_df = static_features(feats).merge(sm[["serial_no", "generation"]], on="serial_no")
    save(h1(sm), "H1_generation_by_region.csv")
    save(h2(sm), "H2_generation_clustering.csv")
    save(h3(feat_df, feats), "H3_generation_morphology.csv", "%.4g")
    a, b = h4_h5(sm, feat_df, feats)
    save(a, "H4_generation_error.csv"); save(b, "H5_generation_offset.csv")
    pd.set_option("display.width", 220)
    print(pd.read_csv(NUMBERS / "H1_generation_by_region.csv").to_string(index=False))
    print(pd.read_csv(NUMBERS / "H2_generation_clustering.csv").to_string(index=False))
    print(pd.read_csv(NUMBERS / "H3_generation_morphology.csv")[["Feature", "Mean_OC3", "Mean_V02Q", "SMD_OC3_minus_V02Q", "MannWhitney_p"]].to_string(index=False, float_format="%.3g"))
    print(a[["Scheme", "Generation", "N_sensors", "RMSE", "MAE", "MBE", "MAE_CI_lo", "MAE_CI_hi"]].to_string(index=False, float_format="%.3f"))
    print(b[["Scheme", "Quantity", "Comparison", "Difference_C", "CI_lo", "CI_hi", "Mean_V02Q", "Mean_OC3"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
