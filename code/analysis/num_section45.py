"""Section 4.5 decomposition by district (G1, G2). Needs extra_inputs/."""

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import stats

from numlib import NUMBERS, SEOUL, TRAIN, save

HM = SEOUL / "04_heat_map"
GU_SHP = HM / "BND_SIGUNGU_PG 2/BND_SIGUNGU_PG.shp"
DATE = "2025-07-08"
HOURS = (4, 14)
MIN_SDOT = 3
KOR_TO_ENG = {"종로구": "Jongno", "중구": "Jung", "용산구": "Yongsan", "성동구": "Seongdong", "광진구": "Gwangjin",
              "동대문구": "Dongdaemun", "중랑구": "Jungnang", "성북구": "Seongbuk", "강북구": "Gangbuk", "도봉구": "Dobong",
              "노원구": "Nowon", "은평구": "Eunpyeong", "서대문구": "Seodaemun", "마포구": "Mapo", "양천구": "Yangcheon",
              "강서구": "Gangseo", "구로구": "Guro", "금천구": "Geumcheon", "영등포구": "Yeongdeungpo", "동작구": "Dongjak",
              "관악구": "Gwanak", "서초구": "Seocho", "강남구": "Gangnam", "송파구": "Songpa", "강동구": "Gangdong"}


def w2(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(np.hypot(a.mean() - b.mean(), a.std(ddof=1) - b.std(ddof=1)))


def districts():
    gu = gpd.read_file(GU_SHP)
    cd = next(c for c in gu.columns if "SIGUNGU_CD" in c.upper() or "SIG_CD" in c.upper())
    nm = next(c for c in gu.columns if "SIGUNGU_NM" in c.upper() or "SIG_KOR_NM" in c.upper() or "SIG_NM" in c.upper())
    s = gu[gu[cd].astype(str).str.startswith("11")].copy()
    s = s.to_crs("EPSG:5179") if s.crs and str(s.crs) != "EPSG:5179" else s
    s["gu_id"] = range(1, len(s) + 1)
    s["name"] = s[nm].map(KOR_TO_ENG).fillna(s[nm])
    s["area_km2"] = s.geometry.area / 1e6
    return s


def label_points(seoul, x, y, res=5.0):
    bb = seoul.total_bounds
    xmin, ymin, xmax, ymax = bb[0] - 50, bb[1] - 50, bb[2] + 50, bb[3] + 50
    nc, nr = int(np.ceil((xmax - xmin) / res)), int(np.ceil((ymax - ymin) / res))
    tf = from_origin(xmin, ymax, res, res)
    grid = rasterize([(g, i) for g, i in zip(seoul.geometry, seoul.gu_id)], out_shape=(nr, nc),
                     transform=tf, fill=0, dtype="int32")
    cf, rf = ~tf * (np.asarray(x), np.asarray(y))
    return grid[np.clip(np.round(rf).astype(int), 0, nr - 1), np.clip(np.round(cf).astype(int), 0, nc - 1)]


def main():
    seoul = districts()
    pred = pl.read_parquet(TRAIN / "Full_Model/Temporal_Holdout/Predictions_2025.parquet",
                           columns=["datetime", "serial_no", "lat", "lon", "actual_temp", "pred_temp"]).to_pandas()
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    corr = []
    for hour in HOURS:
        tag = f"{DATE.replace('-', '')}_{hour:02d}00"
        hm = pl.read_parquet(HM / f"full_heatmaps/parquets/seoul_uhi_{tag}_5m.parquet").to_pandas()
        hm["pred_temp"] = hm.aws_baseline + hm.predicted_residual
        hm["gu_id"] = label_points(seoul, hm.x.values, hm.y.values)
        s = pred[pred.datetime == pd.Timestamp(f"{DATE} {hour:02d}:00")].copy()
        sx, sy = tr.transform(s.lon.values, s.lat.values)
        s["gu_id"] = label_points(seoul, sx, sy)
        rows = []
        for _, d in seoul.iterrows():
            f = hm.loc[hm.gu_id == d.gu_id, "pred_temp"].to_numpy()
            ss = s[s.gu_id == d.gu_id]
            ps, ob = ss.pred_temp.to_numpy(np.float64), ss.actual_temp.to_numpy(np.float64)
            ok = len(ss) >= MIN_SDOT
            rows.append({
                "Hour": f"{hour:02d}:00", "District": d["name"], "N_sensors": len(ss), "Area_km2": d.area_km2,
                "Outdoor_area_km2": len(f) * 25 / 1e6, "Density_per_km2": len(ss) / d.area_km2,
                "Density_per_outdoor_km2": len(ss) / (len(f) * 25 / 1e6) if len(f) else np.nan,
                "Field_mu": f.mean(), "Field_sigma": f.std(ddof=1),
                "PredAtSensors_mu": ps.mean() if len(ps) else np.nan,
                "PredAtSensors_sigma": ps.std(ddof=1) if len(ps) > 1 else np.nan,
                "Obs_mu": ob.mean() if len(ob) else np.nan, "Obs_sigma": ob.std(ddof=1) if len(ob) > 1 else np.nan,
                "W2_field_vs_obs (published)": w2(f, ob) if ok else np.nan,
                "W2_field_vs_pred_at_sensors (placement)": w2(f, ps) if ok else np.nan,
                "W2_pred_vs_obs_at_sensors (model error)": w2(ps, ob) if ok else np.nan,
                "MAE_at_sensors": np.mean(np.abs(ps - ob)) if len(ss) else np.nan,
                "MBE_at_sensors": np.mean(ps - ob) if len(ss) else np.nan,
            })
        df = pd.DataFrame(rows).sort_values("W2_field_vs_obs (published)", ascending=False)
        save(df, f"G1_district_decomposition_{hour:02d}00.csv")
        print(f"\n{hour:02d}:00  sensors matched: {int((s.gu_id > 0).sum())} of {len(s)}")
        print(df[["District", "N_sensors", "Density_per_km2", "W2_field_vs_obs (published)",
                  "W2_field_vs_pred_at_sensors (placement)", "W2_pred_vs_obs_at_sensors (model error)",
                  "MBE_at_sensors"]].to_string(index=False, float_format="%.2f"))
        v = df[df.N_sensors >= MIN_SDOT]
        for wcol in ["W2_field_vs_obs (published)", "W2_field_vs_pred_at_sensors (placement)",
                     "W2_pred_vs_obs_at_sensors (model error)"]:
            for xcol in ["N_sensors", "Density_per_km2", "Density_per_outdoor_km2"]:
                corr.append({"Hour": f"{hour:02d}:00", "W2": wcol, "Against": xcol, "N_districts": len(v),
                             "Pearson_r": stats.pearsonr(v[xcol], v[wcol])[0],
                             "Pearson_p": stats.pearsonr(v[xcol], v[wcol])[1],
                             "Spearman_rho": stats.spearmanr(v[xcol], v[wcol])[0],
                             "Spearman_p": stats.spearmanr(v[xcol], v[wcol])[1]})
        del hm
    c = pd.DataFrame(corr)
    save(c, "G2_district_correlations.csv")
    print("\n", c.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
