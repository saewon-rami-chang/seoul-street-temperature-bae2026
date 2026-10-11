"""SHAP categories (S1, S2), counts and missing inputs (M1-M4), seasonal climate (C1, needs extra_inputs/)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from numlib import NUMBERS, PRED, SEOUL, TRAIN, MONTH_TO_SEASON, cache_units, save, seoul_data

CACHE = SEOUL / "02_cache/chunks"
LAG = {"aws_temp_lag_1h", "aws_temp_lag_2h", "aws_temp_lag_3h", "aws_temp_lag_6h", "aws_temp_mean_6h",
       "aws_temp_mean_24h", "aws_temp_trend_1h", "aws_temp_trend_3h", "aws_temp_trend_24h", "daily_temp_swing",
       "wind_u_mean_3h", "wind_u_mean_6h", "wind_v_mean_3h", "wind_v_mean_6h", "solar_sum_6h", "solar_sum_12h",
       "cloud_mean_3h", "cloud_mean_24h", "precip_sum_3h", "precip_sum_24h", "humidity_mean_3h",
       "humidity_mean_12h", "dew_point_mean_3h", "dew_point_mean_12h", "pressure_trend_3h"}  # Table A.4
TEMPORAL = {"diurnal_phase", "calendar_phase", "solar_elev", "ghi_clearsky"}             # Table 2
RES = ("_10m", "_100m", "_250m", "_1000m")


def category(f):
    if f in LAG: return "Rolling/Lag"
    if f in TEMPORAL: return "Temporal"
    if f.startswith("shadow_fraction_"): return "Shadow"
    if f.startswith("LC_"): return "Land Cover"
    if f.endswith(RES): return "Morphology"
    return "Meteorological"


def climate():
    a = pl.read_parquet(SEOUL / "01_preprocess/03_asos/asos_clean.parquet",
                        columns=["station_id", "datetime", "temperature", "humidity", "wind_speed",
                                 "precipitation", "solar_radiation"]).to_pandas()
    a = a[(a.station_id.astype(str) == "108")]
    a["datetime"] = pd.to_datetime(a["datetime"])
    a = a[a.datetime.dt.year.between(2023, 2025)]
    a["date"] = a.datetime.dt.date
    a["season"] = a.datetime.dt.month.map(MONTH_TO_SEASON)
    daily = a.groupby("date").agg(tmax=("temperature", "max"), tmin=("temperature", "min"),
                                  precip=("precipitation", "sum"), season=("season", "first"),
                                  year=("datetime", lambda s: s.dt.year.iloc[0]))
    rows = []
    for s in ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]:
        h = a[a.season == s]; d = daily[daily.season == s]
        rows.append({"Season": s, "Mean_T_C": h.temperature.mean(), "Mean_daily_Tmax_C": d.tmax.mean(),
                     "Mean_daily_Tmin_C": d.tmin.mean(), "Mean_RH_pct": h.humidity.mean(),
                     "Mean_wind_ms": h.wind_speed.mean(),
                     "Precip_mm_per_season_mean_over_years": d.groupby("year").precip.sum().mean(),
                     "Hours": len(h)})
    df = pd.DataFrame(rows)
    df.insert(0, "Source", "ASOS 108 Seoul, 2023-01-01 to 2025-12-31 (hourly, QC'd)")
    return df


def shap():
    d = pd.read_csv(TRAIN / "Full_Model/SHAP/SHAP_Full.csv")
    d["Category"] = d.Feature.map(category)
    d["Share_pct"] = d.Mean_Abs_SHAP / d.Mean_Abs_SHAP.sum() * 100
    cat = d.groupby("Category").agg(N_features=("Feature", "size"), Share_pct=("Share_pct", "sum")).reset_index()
    text = {"Rolling/Lag": 42.4, "Morphology": 23.2, "Temporal": 17.7, "Meteorological": 8.4, "Shadow": 4.3, "Land Cover": 4.0}
    fig9a = {"Rolling/Lag": 26.1, "Morphology": 23.2, "Temporal": 16.0, "Meteorological": 1.7, "Shadow": 4.3, "Land Cover": 4.0}
    cat["Manuscript_text_pct"] = cat.Category.map(text)
    cat["Published_Fig9a_pct"] = cat.Category.map(fig9a)
    cat["Per_feature_pct"] = cat.Share_pct / cat.N_features
    cat = cat.sort_values("Share_pct", ascending=False)
    note = pd.DataFrame([{"Category": "NOTE", "N_features": np.nan,
                          "Share_pct": np.nan,
                          "Manuscript_text_pct": np.nan,
                          "Published_Fig9a_pct": 100 - sum(fig9a.values())}])
    order = list(cat.Category)
    colors = {"Rolling/Lag": "#2166AC", "Morphology": "#B2182B", "Temporal": "#4DAF4A", "Shadow": "#666666",
              "Land Cover": "#984EA3", "Meteorological": "#FF7F00"}
    fig, ax = plt.subplots(figsize=(3.6, 4))
    bars = ax.bar(range(len(cat)), cat.Share_pct, color=[colors[c] for c in order], edgecolor="k", lw=0.5)
    for b, v in zip(bars, cat.Share_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(cat))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Share of total mean |SHAP| (%)"); ax.set_title("(a) SHAP importance by feature category", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.close(fig)  # figure: full corrected Fig. 9 is drawn by fig09_shap_corrected.py
    return pd.concat([cat, note]), d[["Feature", "Category", "Mean_Abs_SHAP", "Share_pct"]]


def misc():
    meta = json.load(open(TRAIN / "Full_Model/model_metadata.json"))
    hp = pd.DataFrame([{"Parameter": k, "Used_value": v} for k, v in meta["xgb_params"].items()])
    table3 = {"objective": "MAE", "max_depth": 12, "learning_rate": 0.05, "colsample_bytree": 0.55,
              "subsample": 0.8, "min_child_weight": 10}
    hp["Manuscript_Table3"] = hp.Parameter.map(table3)
    hp["Note"] = np.where(hp.Manuscript_Table3.notna() & (hp.Manuscript_Table3.astype(str) != hp.Used_value.astype(str))
                          & ~hp.Parameter.eq("objective"), "MISMATCH", "")
    hp.loc[hp.Parameter == "gamma", "Note"] = "not listed in Table 3"
    hp = pd.concat([hp, pd.DataFrame([{"Parameter": "boosting rounds", "Used_value":
                    f"3-fold CV, early stop patience 40 / min gain 1e-4 over 25 rounds; temporal model {meta['n_rounds']}"}])])

    cv = pd.read_csv(TRAIN / "Discovery_Outputs/Convergence/CV_Fold_Summary.csv")
    master = cv[cv.Fold == "Master"][["Region", "Best_Round", "N_Train_Sensors"]]
    folds = cv[cv.Fold != "Master"].groupby("Region").agg(Fold_rounds=("Best_Round", lambda s: list(map(int, s))),
                                                         Mean_fold_val_MAE=("Val_MAE", "mean")).reset_index()
    sm = pd.read_csv(NUMBERS / "H1_generation_by_region.csv")
    ho = sm[sm.Zone == "Holdout"].set_index("Region").N; bu = sm[sm.Zone == "Buffer"].set_index("Region").N
    lab = {"A_Mapo": "A: Mapo", "B_Gangnam": "B: Gangnam", "C_Central": "C: Central", "D_Gangdong": "D: Gangdong",
           "E_Dobong": "E: Dobong", "F_Gwanak": "F: Gwanak"}
    tc = master.merge(folds, on="Region")
    tc["Holdout_sensors"] = tc.Region.map(lab).map(ho); tc["Buffer_sensors"] = tc.Region.map(lab).map(bu)
    tc = tc.rename(columns={"Best_Round": "Final_rounds", "N_Train_Sensors": "Training_sensors"})
    tc["Total"] = tc.Holdout_sensors + tc.Buffer_sensors + tc.Training_sensors

    files = cache_units()
    feats = meta["features"]
    nonsp = [f for f in feats if not f.endswith(RES)]
    lf = seoul_data.scan(files).filter(pl.col("target_resid").is_not_null() & (pl.col("target_resid").abs() < 25))
    n = lf.select(pl.len()).collect().item()
    # KMA leaves solar radiation empty at night, so also report daylight-only rates (solar elevation > 0)
    agg = lambda frame: frame.select([pl.col(f).null_count().alias(f) for f in nonsp] +
                                     [pl.any_horizontal([pl.col(f).is_null() for f in nonsp]).sum().alias("__any"),
                                      pl.len().alias("__n")]).collect().to_dicts()[0]
    nn, nd = agg(lf), agg(lf.filter(pl.col("solar_elev") > 0))
    miss = pd.DataFrame([{"Feature": f, "Category": category(f), "Missing_pct_all_hours": nn[f] / n * 100,
                          "Missing_pct_daylight": nd[f] / nd["__n"] * 100} for f in nonsp])
    miss = miss.sort_values("Missing_pct_daylight", ascending=False)
    miss = pd.concat([miss, pd.DataFrame([
        {"Feature": "ANY of the 38 non-spatial inputs", "Missing_pct_all_hours": nn["__any"] / n * 100,
         "Missing_pct_daylight": nd["__any"] / nd["__n"] * 100},
        {"Feature": "NOTE: solar_radiation is empty at night by KMA convention (100 % missing 21-05 h)"},
        {"Feature": "rows (2023-2025, |r|<25)", "Missing_pct_all_hours": n, "Missing_pct_daylight": nd["__n"]}])])

    counts = [{"Count": 1217, "What": "sensors in raw S-DoT download (Table A.1 stage 1)", "Source": "manuscript Table A.1"},
              {"Count": 1146, "What": "sensors after QC (Table A.1 final)", "Source": "manuscript Table A.1"}]
    sids = seoul_data.scan(files).select("serial_no").unique().collect().height
    sids_valid = lf.select("serial_no").unique().collect().height
    for y in (2023, 2024, 2025):
        fy = [u for u in files if u.year == y]
        counts.append({"Count": seoul_data.scan(fy).filter(pl.col("target_resid").is_not_null()).select("serial_no")
                       .unique().collect().height, "What": f"sensors with >=1 valid target in {y}", "Source": "cache"})
    counts += [{"Count": sids, "What": "sensors present in 2023-2025 cache (LORO sensor map)", "Source": "cache"},
               {"Count": sids_valid, "What": "sensors with >=1 valid target 2023-2025", "Source": "cache"},
               {"Count": 1106, "What": "sensors in temporal-model training data (2023-2024)", "Source": "Full_Model CV_Fold_Summary"},
               {"Count": 1112, "What": "sensors evaluated in 2025 temporal holdout (Table 7 says 1,111)", "Source": "Temporal_Holdout Metrics"},
               {"Count": len(pd.read_csv(TRAIN / "Full_Model/Temporal_Holdout/PerSensor_Performance.csv")),
                "What": "2025 temporal-holdout sensors with >=24 valid hours (PerSensor_Performance.csv) = the manuscript's 1,111; "
                        "also the n in Fig. 1 (07_draw_loro.py reads this file)", "Source": "Full_Model/Temporal_Holdout"},
               {"Count": 225, "What": "test sensors in random 80/20 holdout (text says 222 = 20 % of 1,111)", "Source": "Random8020 predictions"},
               {"Count": sids_valid - 225, "What": "training sensors in random 80/20 split (text says 889 = 1,111 - 222)", "Source": "derived"},
               {"Count": 370, "What": "LORO holdout sensors (Table 6)", "Source": "Discovery_Outputs"}]
    return hp, tc, miss, pd.DataFrame(counts)


def main():
    save(climate(), "C1_seasonal_climate_ASOS108.csv", "%.2f")
    s1, s2 = shap()
    save(s1, "S1_shap_categories.csv", "%.2f"); save(s2, "S2_shap_feature_category_map.csv")
    hp, tc, miss, counts = misc()
    save(hp, "M1_hyperparameters_used.csv"); save(tc, "M2_loro_training_counts.csv")
    save(miss, "M3_missing_inputs.csv", "%.3f"); save(counts, "M4_sensor_counts.csv")
    pd.set_option("display.width", 200)
    for f in ["C1_seasonal_climate_ASOS108.csv", "S1_shap_categories.csv", "M1_hyperparameters_used.csv",
              "M2_loro_training_counts.csv", "M4_sensor_counts.csv"]:
        print(f"\n{f}\n", pd.read_csv(NUMBERS / f).to_string(index=False))
    print("\nM3 (top 8 + any):\n", pd.read_csv(NUMBERS / "M3_missing_inputs.csv").head(8).to_string(index=False),
          "\n", pd.read_csv(NUMBERS / "M3_missing_inputs.csv").tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
