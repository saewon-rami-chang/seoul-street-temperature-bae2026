"""Feature-space coverage of the LORO regions vs the city grid (F0-F2). Needs extra_inputs/."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import rasterio
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.stats import wasserstein_distance

from numlib import NUMBERS, SEOUL, TRAIN, REGION_LABEL, save, scan_cache

GRID = SEOUL / "04_heat_map/full_heatmaps/parquets/seoul_uhi_20250708_0400_5m.parquet"
MORPH = SEOUL / "01_preprocess/04_morphology/processed"
LC = SEOUL / "01_preprocess/05_landcover/processed"
N_GRID = 50_000
N_TOP = 15
rng = np.random.default_rng(42)

# land-cover super-classes, as in the feature pipeline
LC_SUPERCLASS = {
    "LC_Urban": {111, 112, 121, 131, 132, 141, 151, 152, 153, 154, 155, 161, 162, 163},
    "LC_Agri": {211, 212, 221, 222, 231, 241, 251, 252},
    "LC_Forest": {311, 321, 331}, "LC_Grass": {411, 421, 422, 423},
    "LC_Water": {511, 521, 611, 612, 711, 712}, "LC_Barren": {621, 622, 623},
}


def bilinear(path, band_idx, x, y, nodata):
    """Vectorized copy of build_cache.bilinear_sample for selected bands (0-based)."""
    with rasterio.open(path) as src:
        arr = src.read([i + 1 for i in band_idx]).astype(np.float64)
        inv = ~src.transform
    if nodata is not None:
        arr[arr == nodata] = np.nan
    h, w = arr.shape[1:]
    cf, rf = inv * (x, y)
    r0 = np.floor(rf).astype(int); c0 = np.floor(cf).astype(int)
    dr = rf - r0; dc = cf - c0
    r1 = r0 + 1; c1 = c0 + 1
    m = r0 < 0; r0[m], r1[m], dr[m] = 0, 1, 0.0
    m = c0 < 0; c0[m], c1[m], dc[m] = 0, 1, 0.0
    m = r1 >= h; r0[m], r1[m], dr[m] = h - 2, h - 1, 1.0
    m = c1 >= w; c0[m], c1[m], dc[m] = w - 2, w - 1, 1.0
    corners = np.stack([arr[:, r0, c0], arr[:, r0, c1], arr[:, r1, c0], arr[:, r1, c1]], axis=-1)
    wts = np.stack([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc], axis=-1)[None]
    valid = ~np.isnan(corners)
    wv = wts * valid
    s = wv.sum(-1)
    out = np.where(s > 0, (np.where(valid, corners, 0) * wv).sum(-1) / np.where(s > 0, s, 1), np.nan)
    return out.T  # (n_points, n_bands)


def sample_features(feats, x, y):
    out = pd.DataFrame(index=range(len(x)))
    by_file = {}
    for f in feats:
        name, res = f.rsplit("_", 1)
        by_file.setdefault(res, []).append((f, name))
    for res, items in by_file.items():
        morph = [(f, n) for f, n in items if not n.startswith("LC_")]
        lc = [(f, n) for f, n in items if n.startswith("LC_")]
        if morph:
            with rasterio.open(MORPH / f"Seoul_{res}.tif") as s:
                desc = list(s.descriptions)
            vals = bilinear(MORPH / f"Seoul_{res}.tif", [desc.index(n) for _, n in morph], x, y, -9999)
            for j, (f, _) in enumerate(morph):
                out[f] = vals[:, j]
        for f, n in lc:
            p = LC / f"LandCover_Fractions_{res}.tif"
            with rasterio.open(p) as s:
                codes = [int(d.split("_")[0]) for d in s.descriptions]
            idx = [i for i, c in enumerate(codes) if c in LC_SUPERCLASS[n]]
            out[f] = np.nansum(bilinear(p, idx, x, y, -1), axis=1)
    return out


def top_features():
    d = pd.read_csv(TRAIN / "Full_Model/SHAP/SHAP_Full.csv")
    sp = d[d.Feature.str.match(r".*_(?:10|100|250|1000)m$") & ~d.Feature.str.startswith("shadow_fraction")]
    return sp.nlargest(N_TOP, "Mean_Abs_SHAP")["Feature"].tolist()


def main():
    feats = top_features()
    sm = pd.read_csv(TRAIN / "Discovery_Outputs/Spatial/Sensor_Map.csv")
    bounds = json.load(open(TRAIN / "Discovery_Outputs/Spatial/Region_Bounds_5179.json"))
    sm["region"] = "-"
    for k, b in bounds.items():
        sm.loc[sm.x.between(b["x_min"], b["x_max"]) & sm.y.between(b["y_min"], b["y_max"]), "region"] = REGION_LABEL[k]

    cache = (scan_cache().select(["serial_no", *feats]).group_by("serial_no")
             .agg([pl.col(f).drop_nulls().first() for f in feats]).collect().to_pandas())
    mine = sample_features(feats, sm.x.to_numpy(np.float64), sm.y.to_numpy(np.float64))
    mine["serial_no"] = sm.serial_no.values
    chk = cache.merge(mine, on="serial_no", suffixes=("_cache", "_sampler"))
    f0 = pd.DataFrame([{"Feature": f, "Max_abs_diff": np.nanmax(np.abs(chk[f + "_cache"] - chk[f + "_sampler"])),
                        "Median_abs_diff": np.nanmedian(np.abs(chk[f + "_cache"] - chk[f + "_sampler"])),
                        "NaN_mismatch": int((chk[f + "_cache"].isna() != chk[f + "_sampler"].isna()).sum()),
                        "Feature_SD": chk[f + "_cache"].std()} for f in feats])
    save(f0, "F0_sampler_check.csv", "%.6g")
    print(f0.to_string(index=False))
    sens = cache.merge(sm[["serial_no", "region", "x", "y"]], on="serial_no")  # use cache values for sensors

    g = pl.read_parquet(GRID, columns=["x", "y"])
    idx = rng.choice(g.height, N_GRID, replace=False)
    gx = g["x"].to_numpy()[idx].astype(np.float64); gy = g["y"].to_numpy()[idx].astype(np.float64)
    print(f"grid: {g.height:,} outdoor points, sampled {N_GRID:,}")
    grid = sample_features(feats, gx, gy)

    groups = {lab: sens[sens.region == lab] for lab in REGION_LABEL.values()}
    groups["All 6 holdout regions (370)"] = sens[sens.region != "-"]
    groups[f"All sensors ({len(sens)})"] = sens
    rows = []
    for f in feats:
        gv = grid[f].dropna().to_numpy(); sd = gv.std()
        for lab, s in groups.items():
            v = s[f].dropna().to_numpy()
            rows.append({"Feature": f, "Group": lab, "N": len(v),
                         "W1_standardized": wasserstein_distance(gv, v) / sd if sd > 0 else np.nan,
                         "Grid_pct_inside_range": np.mean((gv >= v.min()) & (gv <= v.max())) * 100,
                         "Grid_pct_inside_P5_P95": np.mean((gv >= np.percentile(v, 5)) & (gv <= np.percentile(v, 95))) * 100,
                         "Grid_mean": gv.mean(), "Group_mean": v.mean(), "Grid_median": np.median(gv),
                         "Group_median": np.median(v)})
    f1 = pd.DataFrame(rows)
    save(f1, "F1_coverage_per_feature.csv")

    G = grid[feats].dropna()
    mu, sd = G.mean(), G.std()
    Z = ((G - mu) / sd).to_numpy()
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    expl = S ** 2 / np.sum(S ** 2)
    pcs = lambda df: ((df[feats] - mu) / sd).to_numpy() @ Vt[:2].T
    gp = Z @ Vt[:2].T
    cov_inv = np.linalg.inv(np.cov(Z, rowvar=False))
    rows = []
    for lab, s in groups.items():
        sv = s[feats].dropna()
        sp = pcs(sv)
        hull = ConvexHull(sp)
        inside = MplPath(sp[hull.vertices]).contains_points(gp)
        zs = ((sv - mu) / sd).to_numpy()
        d_s = np.einsum("ij,jk,ik->i", zs, cov_inv, zs)
        d_g = np.einsum("ij,jk,ik->i", Z, cov_inv, Z)
        rows.append({"Group": lab, "N_sensors": len(sv), "PC1_PC2_var_explained_pct": expl[:2].sum() * 100,
                     "Grid_pct_inside_sensor_hull_PC1_PC2": inside.mean() * 100,
                     "Grid_pct_within_sensor_P95_Mahalanobis": np.mean(d_g <= np.percentile(d_s, 95)) * 100})
    f2 = pd.DataFrame(rows)
    save(f2, "F2_coverage_multivariate.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(gp[:, 0], gp[:, 1], gridsize=70, bins="log", cmap="Greys", mincnt=1)
    allp = pcs(sens[feats].dropna())
    hp = pcs(groups["All 6 holdout regions (370)"][feats].dropna())
    ax.scatter(allp[:, 0], allp[:, 1], s=5, c="#4C72B0", label=f"All sensors (n={len(allp)})")
    ax.scatter(hp[:, 0], hp[:, 1], s=7, c="#DD8452", label=f"LORO holdout sensors (n={len(hp)})")
    ax.set_xlabel(f"PC1 ({expl[0]*100:.0f} %)"); ax.set_ylabel(f"PC2 ({expl[1]*100:.0f} %)")
    ax.set_title("Top-15 spatial features: Seoul outdoor grid (grey) vs sensors", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(NUMBERS / "figures/F_coverage_pca.png", dpi=300)
    fig.savefig(NUMBERS / "figures/F_coverage_pca.pdf", dpi=600)
    fig.savefig(NUMBERS / "figures/F_coverage_pca.svg")
    plt.close(fig)
    print("  -> numbers/figures/F_coverage_pca.png/.pdf/.svg")

    pd.set_option("display.width", 200)
    piv = f1.pivot_table(index="Feature", columns="Group", values="W1_standardized")
    print("\nStandardized W1 to Seoul grid:\n", piv.round(2).to_string())
    print("\n", f2.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
