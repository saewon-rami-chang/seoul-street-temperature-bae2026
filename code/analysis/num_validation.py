"""Tables 5-7 with 95 % sensor-bootstrap CIs; day/night, hourly, seasonal (V1-V4)."""

import numpy as np
import pandas as pd

from numlib import PRED, REGION_LABEL, MONTH_TO_SEASON, load_pred, metrics, sensor_ci, save

DAY_NIGHT = {  # definition -> (label, boolean mask function on hour)
    "06-17 day / 18-05 night": lambda h: (h >= 6) & (h <= 17),
    "07-18 day / 19-06 night": lambda h: (h >= 7) & (h <= 18),
    "12-17 day / 00-05 night (Paper 2 windows)": None,  # handled separately
    "Manuscript nocturnal 21-05": None,
}


def rows_for(df, scheme, group_name, mask=None, strata=None, rng=None):
    sub = df if mask is None else df[mask]
    out = []
    for model, col in (("XGBoost", "pred_temp"), ("AWS baseline", "aws_temp_interp")):
        r = {"Scheme": scheme, "Subset": group_name, "Model": model, **metrics(sub["actual_temp"], sub[col])}
        if model == "XGBoost":
            r["R2_residual"] = metrics(sub["target_resid"], sub["pred_resid"])["R2"]
        r.update(sensor_ci(sub, col, strata=strata, rng=rng))
        out.append(r)
    return out


def main():
    rng = np.random.default_rng(42)
    v1, v2, v3, v4 = [], [], [], []
    for scheme, path in PRED.items():
        print(f"{scheme} ...")
        extra = ("region",) if scheme != "Temporal 2025" else ()
        df = load_pred(path, extra)
        h = df["datetime"].dt.hour
        strata = "region" if scheme == "Spatial LORO" else None

        # V1: overall (+ per region for LORO = Table 6)
        v1 += rows_for(df, scheme, "All", strata=strata, rng=rng)
        if scheme == "Spatial LORO":
            for key, lab in REGION_LABEL.items():
                v1 += rows_for(df, scheme, lab, df["region"] == key, rng=rng)

        # V2: day / night
        defs = {
            "Day 06:00-17:59": (h >= 6) & (h <= 17), "Night 18:00-05:59": (h >= 18) | (h <= 5),
            "Day 12:00-17:59": (h >= 12) & (h <= 17), "Night 00:00-05:59": h <= 5,
            "Nocturnal 21:00-05:59 (Sec. 4.1)": (h >= 21) | (h <= 5), "Pre-dawn 02:00-05:59": (h >= 2) & (h <= 5),
        }
        for name, m in defs.items():
            v2 += rows_for(df, scheme, name, m, strata=strata, rng=rng)

        # V3: hourly (no CI; 24 x 3 schemes)
        e = df["pred_temp"] - df["actual_temp"]
        for hour in range(24):
            m = h == hour
            r = metrics(df.loc[m, "actual_temp"], df.loc[m, "pred_temp"])
            r.update({"Scheme": scheme, "Hour": hour, "Median_AE": float(e[m].abs().median()),
                      "Baseline_MAE": metrics(df.loc[m, "actual_temp"], df.loc[m, "aws_temp_interp"])["MAE"]})
            v3.append(r)

        # V4: seasons
        season = df["datetime"].dt.month.map(MONTH_TO_SEASON)
        for s in ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]:
            v4 += rows_for(df, scheme, s, season == s, strata=strata, rng=rng)
        del df

    save(pd.DataFrame(v1), "V1_validation_tables_CI.csv")
    save(pd.DataFrame(v2), "V2_day_night.csv")
    v3 = pd.DataFrame(v3)[["Scheme", "Hour", "N_obs", "MAE", "Median_AE", "RMSE", "MBE", "R2", "Baseline_MAE"]]
    save(v3, "V3_hourly.csv")
    save(pd.DataFrame(v4), "V4_seasonal.csv")

    show = pd.DataFrame(v1)
    print(show[show.Model == "XGBoost"][["Scheme", "Subset", "N_sensors", "RMSE", "RMSE_CI_lo", "RMSE_CI_hi",
                                         "MAE", "MAE_CI_lo", "MAE_CI_hi", "MBE", "R2", "R2_residual"]]
          .to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
