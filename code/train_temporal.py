"""Temporal holdout: train 2023-2024, test 2025."""

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["POLARS_MAX_THREADS"] = "8"

import argparse
import gc
import json
import logging
import subprocess
import numpy as np
import pandas as pd
import polars as pl
import seoul_data
import xgboost as xgb
from pathlib import Path
from pyproj import Transformer
from sklearn.model_selection import GroupKFold
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

META_COLS = {"serial_no", "datetime", "lat", "lon", "aws_temp_interp", "target_resid"}
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
OUTLIER_THRESHOLD = 25.0
SEED = 42
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring",  5: "Spring",
    6: "Summer",  7: "Summer",  8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


def detect_device():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("GPU detected — using CUDA")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.info("No GPU detected — using CPU")
    return "cpu"


XGB_PARAMS = {
    "objective":        "reg:absoluteerror",
    "eval_metric":      "mae",
    "tree_method":      "hist",
    "max_bin":          256,
    "n_jobs":           8,
    "learning_rate":    0.05,
    "max_depth":        12,
    "min_child_weight": 50,
    "subsample":        0.9,
    "colsample_bytree": 0.55,
    "gamma":            0.7,
    "seed":             SEED,
}

TRAIN_YEARS = [2023, 2024]
TEST_YEAR = 2025


class _ChunkIter(xgb.DataIter):
    def __init__(self, files, features, serial_nos=None, target="target_resid"):
        self._files = files
        self._features = features
        self._sids = serial_nos  # None = all sensors
        self._target = target
        self._it = 0
        super().__init__()

    def next(self, input_data):
        while self._it < len(self._files):
            df = self._files[self._it].read()
            self._it += 1
            filters = (
                pl.col(self._target).is_not_null() &
                (pl.col(self._target).abs() < OUTLIER_THRESHOLD)
            )
            if self._sids is not None:
                filters = filters & pl.col("serial_no").is_in(self._sids)
            df = df.filter(filters)
            if df.height > 0:
                X = df.select(self._features).to_numpy().astype(np.float32)
                y = df[self._target].to_numpy().astype(np.float32)
                input_data(data=X, label=y)
                del df
                return 1
            del df
        return 0

    def reset(self):
        self._it = 0


def _make_cb(patience=40, min_delta=0.0001, window=25):
    """Early stopping: patience OR negligible per-round gain over window."""
    class _ES(xgb.callback.TrainingCallback):
        def __init__(self):
            super().__init__()
            self.best_score = float("inf")
            self.best_round = 0
            self.history: list[float] = []
        def after_iteration(self, model, epoch, evals_log):
            try:
                v = evals_log["valid"]["mae"][-1]
            except KeyError:
                return False
            self.history.append(v)
            if epoch % 50 == 0:
                logger.info(f"      [Rnd {epoch:>4}] val={v:.5f} best={self.best_score:.5f}@{self.best_round}")
            if v < self.best_score:
                self.best_score = v; self.best_round = epoch
            if epoch - self.best_round >= patience:
                logger.info(f"      [Stop] Patience @{epoch}. best={self.best_score:.5f}@{self.best_round}")
                return True
            if len(self.history) >= window + 1:
                g = (self.history[-(window+1)] - self.history[-1]) / window
                if g < min_delta:
                    logger.info(f"      [Stop] MinDelta @{epoch}. best={self.best_score:.5f}@{self.best_round}")
                    return True
            return False
    return _ES()


def compute_metrics(y_true, y_pred, full=True) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true); err = y_pred - y_true; ae = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(ae))
    mbe  = float(np.mean(err))
    ss_r = np.sum(err**2); ss_t = np.sum((y_true - y_true.mean())**2)
    r2 = float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")
    out = {"RMSE": rmse, "MAE": mae, "MBE": mbe, "R2": r2, "N": int(n)}
    if not full:
        return out
    rv, _ = scipy_stats.pearsonr(y_true, y_pred)
    out["Pearson_r"] = float(rv)
    denom = np.sum((np.abs(y_pred - y_true.mean()) + np.abs(y_true - y_true.mean()))**2)
    out["Willmott_d"] = float(1 - ss_r / denom) if denom > 0 else float("nan")
    out["Hit_05C"] = float(np.mean(ae <= 0.5))
    out["Hit_10C"] = float(np.mean(ae <= 1.0))
    out["Hit_20C"] = float(np.mean(ae <= 2.0))
    sl, ic = np.polyfit(y_true, y_pred, 1)
    yh = sl * y_true + ic
    out["RMSEs"] = float(np.sqrt(np.mean((yh - y_true)**2)))
    out["RMSEu"] = float(np.sqrt(np.mean((y_pred - yh)**2)))
    out["Error_Std"] = float(np.std(err))
    out["Skewness"]  = float(scipy_stats.skew(err))
    out["Kurtosis"]  = float(scipy_stats.kurtosis(err))
    for p, t in [(50,"P50"),(75,"P75"),(90,"P90"),(95,"P95"),(99,"P99")]:
        out[f"AE_{t}"] = float(np.percentile(ae, p))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(SEED)

    cache_dir = seoul_data.default_data_dir()
    out_dir = (seoul_data.default_results_dir() / "Full_Model").resolve()
    holdout_dir = out_dir / "Temporal_Holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    params = {**XGB_PARAMS, "device": device}

    all_files = seoul_data.units(cache_dir)
    train_files = [u for u in all_files if u.year in TRAIN_YEARS]
    test_files = [u for u in all_files if u.year == TEST_YEAR]
    logger.info(f"Train chunks: {len(train_files)} ({TRAIN_YEARS})")
    logger.info(f"Test chunks:  {len(test_files)} ({TEST_YEAR})")

    schema_cols = seoul_data.schema_names(train_files)
    features = sorted(schema_cols - META_COLS)
    logger.info(f"Features: {len(features)}")

    if args.rounds:
        opt_rounds = args.rounds
        src = "manual"
        logger.info(f"  Rounds: {opt_rounds} ({src})")
    else:
        logger.info(f"\n{'='*60}")
        logger.info("3-FOLD CV — discovering optimal round count")
        logger.info(f"{'='*60}")

        logger.info("Collecting sensor IDs for fold splitting...")
        all_sids = set()
        all_sids.update(seoul_data.scan(train_files).select("serial_no").unique().collect()["serial_no"].to_list())
        all_sids = np.array(sorted(all_sids))
        logger.info(f"  {len(all_sids)} unique sensors in training data")

        gkf = GroupKFold(n_splits=3)
        # GroupKFold needs a dummy X/y with one entry per group
        dummy = np.arange(len(all_sids))
        fold_best_rounds = []
        cv_summary_rows = []
        all_evals = []

        for fold_i, (trn_idx, val_idx) in enumerate(gkf.split(dummy, groups=all_sids)):
            trn_sids = set(all_sids[trn_idx])
            val_sids = set(all_sids[val_idx])
            logger.info(f"\n  Fold {fold_i+1}/3 — train {len(trn_sids)} sensors, val {len(val_sids)} sensors")

            logger.info(f"    Building DMatrices...")
            it_trn = _ChunkIter(train_files, features, serial_nos=trn_sids)
            d_trn = xgb.QuantileDMatrix(it_trn, max_bin=256)
            it_val = _ChunkIter(train_files, features, serial_nos=val_sids)
            d_val = xgb.QuantileDMatrix(it_val, ref=d_trn, max_bin=256)

            er: dict = {}
            cb = _make_cb()
            xgb.train(
                params, d_trn, num_boost_round=3000,
                evals=[(d_trn, "train"), (d_val, "valid")],
                evals_result=er, callbacks=[cb], verbose_eval=False,
            )

            cv_summary_rows.append({
                "Fold": fold_i + 1,
                "Best_Round": cb.best_round, "Val_MAE": cb.best_score,
                "N_Train_Sensors": len(trn_sids), "N_Val_Sensors": len(val_sids),
            })
            fold_best_rounds.append(cb.best_round)
            all_evals.append(er)
            logger.info(f"    Fold {fold_i+1} -> rnd={cb.best_round} MAE={cb.best_score:.5f}")
            del d_trn, d_val, it_trn, it_val; gc.collect()

        arr = np.array(fold_best_rounds)
        cv_coeff = arr.std() / arr.mean() if arr.mean() > 0 else 0
        opt_rounds = int(np.median(arr)) if cv_coeff > 0.30 else int(np.mean(arr))
        method = "median" if cv_coeff > 0.30 else "mean"
        logger.info(f"\n  Rounds: {list(arr)} -> {method}={opt_rounds} (CV={cv_coeff:.2f})")
        src = f"3-fold CV {method} (per-fold: {list(arr)})"

        cv_summary_rows.append({
            "Fold": "Master",
            "Best_Round": opt_rounds, "Val_MAE": np.nan,
            "N_Train_Sensors": len(all_sids), "N_Val_Sensors": 0,
        })

        convergence_dir = out_dir / "Convergence"
        convergence_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cv_summary_rows).to_csv(
            convergence_dir / "CV_Fold_Summary.csv", index=False)

        conv = []
        for fold_i, er in enumerate(all_evals):
            for sp in ("train", "valid"):
                for rnd, val in enumerate(er[sp]["mae"]):
                    conv.append({"Fold": fold_i + 1,
                                 "Round": rnd, "Split": sp, "MAE": val})
        pd.DataFrame(conv).to_csv(
            convergence_dir / "Convergence_Data.csv", index=False)

        logger.info(f"  CV optimal rounds: {opt_rounds} ({src})")

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING — {opt_rounds} rounds, all sensors, {TRAIN_YEARS}, device={device}")
    logger.info(f"{'='*60}")

    logger.info("Building train DMatrix...")
    it = _ChunkIter(train_files, features)
    dtrain = xgb.QuantileDMatrix(it, max_bin=256)

    logger.info("Training...")
    model = xgb.train(params, dtrain, num_boost_round=opt_rounds,
                      evals=[(dtrain, "train")], verbose_eval=100)
    del dtrain; gc.collect()

    model.feature_names = features

    model.save_model(str(out_dir / "Full_Model.json"))
    logger.info(f"  Model saved: Full_Model.json")

    gain = model.get_score(importance_type="gain")
    wt = model.get_score(importance_type="weight")
    pd.DataFrame({
        "Feature": list(gain.keys()), "Gain": list(gain.values()),
        "Weight": [wt.get(k, 0) for k in gain.keys()],
    }).sort_values("Gain", ascending=False).to_csv(
        out_dir / "Native_Importance_Full.csv", index=False)

    json.dump({
        "features": features, "n_features": len(features),
        "n_rounds": opt_rounds, "round_source": src,
        "train_years": TRAIN_YEARS, "test_year": TEST_YEAR,
        "device": device,
        "xgb_params": {k: v for k, v in params.items() if k != "device"},
    }, open(out_dir / "model_metadata.json", "w"), indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEMPORAL HOLDOUT — predicting {TEST_YEAR}")
    logger.info(f"{'='*60}")

    logger.info("Loading 2025 data...")
    df_test = (
        seoul_data.scan(test_files)
        .filter(
            pl.col("target_resid").is_not_null() &
            (pl.col("target_resid").abs() < OUTLIER_THRESHOLD)
        )
        .with_columns([pl.col(c).cast(pl.Float32) for c in features])
        .collect()
    )
    logger.info(f"  {df_test.height:,} rows, {df_test.select('serial_no').unique().height} sensors")

    X_test = df_test.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X_test, feature_names=features)
    pred_resid = model.predict(dm)
    del dm, X_test; gc.collect()

    aws = df_test["aws_temp_interp"].to_numpy()
    tgt = df_test["target_resid"].to_numpy()
    df_eval = df_test.select(
        ["datetime", "serial_no", "lat", "lon", "aws_temp_interp", "target_resid"]
    ).to_pandas()
    del df_test; gc.collect()

    df_eval["pred_resid"]  = pred_resid
    df_eval["actual_temp"] = (aws + tgt).astype(np.float32)
    df_eval["pred_temp"]   = (aws + pred_resid).astype(np.float32)
    df_eval["Error"]       = df_eval["pred_temp"] - df_eval["actual_temp"]
    df_eval["datetime"]    = pd.to_datetime(df_eval["datetime"])
    df_eval["eval_hour"]   = df_eval["datetime"].dt.hour.astype(int)
    df_eval["eval_month"]  = df_eval["datetime"].dt.month.astype(int)
    df_eval["season"]      = df_eval["eval_month"].map(MONTH_TO_SEASON)

    df_eval.to_parquet(holdout_dir / "Predictions_2025.parquet", index=False)

    logger.info("\n>>> Metrics...")
    ma = compute_metrics(df_eval["actual_temp"], df_eval["pred_temp"], full=True)
    ba = compute_metrics(df_eval["actual_temp"], df_eval["aws_temp_interp"], full=True)
    mr = compute_metrics(df_eval["target_resid"], df_eval["pred_resid"], full=False)
    ns = df_eval["serial_no"].nunique()

    rows_abs = [
        {"Model": "XGBoost",      "N_sensors": ns, **ma},
        {"Model": "AWS Baseline", "N_sensors": ns, **ba},
    ]
    pd.DataFrame(rows_abs).to_csv(holdout_dir / "Metrics_Absolute.csv", index=False)
    pd.DataFrame([{"N_sensors": ns, **mr}]).to_csv(holdout_dir / "Metrics_Residual.csv", index=False)

    logger.info(f"  XGBoost:  MAE={ma['MAE']:.3f}  RMSE={ma['RMSE']:.3f}  R2={ma['R2']:.4f}  MBE={ma['MBE']:+.3f}")
    logger.info(f"  Baseline: MAE={ba['MAE']:.3f}  RMSE={ba['RMSE']:.3f}  R2={ba['R2']:.4f}  MBE={ba['MBE']:+.3f}")
    logger.info(f"  Residual: MAE={mr['MAE']:.3f}  R2={mr['R2']:.4f}")
    logger.info(f"  Improvement: {(ba['MAE']-ma['MAE'])/ba['MAE']*100:.1f}% MAE reduction")

    logger.info(">>> Per-sensor...")
    srows = []
    for sno, g in df_eval.groupby("serial_no"):
        if len(g) < 24: continue
        sm = compute_metrics(g["actual_temp"], g["pred_temp"], full=False)
        sr = compute_metrics(g["target_resid"], g["pred_resid"], full=False)
        srows.append({
            "serial_no": sno, "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0],
            "N_hours": len(g),
            "abs_MAE": sm["MAE"], "abs_RMSE": sm["RMSE"],
            "abs_MBE": sm["MBE"], "abs_R2": sm["R2"],
            "res_MAE": sr["MAE"], "res_RMSE": sr["RMSE"],
            "res_MBE": sr["MBE"], "res_R2": sr["R2"]})
    sensor_df = pd.DataFrame(srows)
    sensor_df.to_csv(holdout_dir / "PerSensor_Performance.csv", index=False)

    summ = []
    for col in ["abs_MAE", "abs_RMSE", "abs_MBE", "res_MAE"]:
        summ.append({"Metric": col, "N": len(sensor_df),
            "Mean": sensor_df[col].mean(), "Std": sensor_df[col].std(),
            "Median": sensor_df[col].median(),
            "Min": sensor_df[col].min(), "Max": sensor_df[col].max()})
    pd.DataFrame(summ).to_csv(holdout_dir / "PerSensor_Summary.csv", index=False)
    logger.info(f"  {len(sensor_df)} sensors")

    logger.info(">>> Baseline improvement...")
    pd.DataFrame([{
        "XGB_MAE": ma["MAE"], "AWS_MAE": ba["MAE"],
        "Delta_MAE": ma["MAE"] - ba["MAE"],
        "Pct_MAE": (ba["MAE"] - ma["MAE"]) / ba["MAE"] * 100,
        "XGB_RMSE": ma["RMSE"], "AWS_RMSE": ba["RMSE"],
        "Delta_RMSE": ma["RMSE"] - ba["RMSE"],
        "Pct_RMSE": (ba["RMSE"] - ma["RMSE"]) / ba["RMSE"] * 100,
    }]).to_csv(holdout_dir / "Baseline_Improvement.csv", index=False)

    logger.info(">>> Temporal stats...")
    df_eval.groupby("eval_hour")["Error"].agg(
        Mean="mean", Median="median", Std="std",
        Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75), N="count"
    ).reset_index().rename(columns={"eval_hour": "Hour"}).to_csv(
        holdout_dir / "Diurnal_Stats.csv", index=False)

    df_eval.groupby("eval_month")["Error"].agg(
        Mean="mean", Median="median", Std="std",
        Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75), N="count"
    ).reset_index().rename(columns={"eval_month": "Month"}).to_csv(
        holdout_dir / "Monthly_Stats.csv", index=False)

    sd = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        sub = df_eval[df_eval["season"] == season]
        if len(sub) == 0: continue
        st = sub.groupby("eval_hour")["Error"].agg(
            Mean="mean", Std="std", N="count",
            Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75)
        ).reset_index()
        st.insert(0, "Season", season)
        sd.append(st)
    pd.concat(sd, ignore_index=True).to_csv(
        holdout_dir / "SeasonDiurnal_Stats.csv", index=False)

    logger.info(">>> Temp-stratified...")
    bins = [-999, 0, 10, 20, 30, 999]
    bin_labels = ["<0C", "0-10C", "10-20C", "20-30C", ">30C"]
    df_eval["temp_bin"] = pd.cut(df_eval["actual_temp"], bins=bins, labels=bin_labels)
    tstrat = []
    for bl in bin_labels:
        sub = df_eval[df_eval["temp_bin"] == bl]
        if len(sub) < 10: continue
        tstrat.append({"TempBin": bl, "Model": "XGBoost",
                       **compute_metrics(sub["actual_temp"], sub["pred_temp"])})
        tstrat.append({"TempBin": bl, "Model": "AWS Baseline",
                       **compute_metrics(sub["actual_temp"], sub["aws_temp_interp"])})
    pd.DataFrame(tstrat).to_csv(holdout_dir / "TempStratified.csv", index=False)

    pd.DataFrame([{
        "MBE": float(df_eval["Error"].mean()), "Std": float(df_eval["Error"].std()),
        "Skew": float(scipy_stats.skew(df_eval["Error"])),
        "Kurt": float(scipy_stats.kurtosis(df_eval["Error"])),
        "Min": float(df_eval["Error"].min()), "Max": float(df_eval["Error"].max()),
        "Hit_05C": float((df_eval["Error"].abs() <= 0.5).mean()),
        "Hit_10C": float((df_eval["Error"].abs() <= 1.0).mean()),
        "Hit_20C": float((df_eval["Error"].abs() <= 2.0).mean()),
    }]).to_csv(holdout_dir / "Error_Distribution.csv", index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE — Full_Model/")
    logger.info(f"  Model:     Full_Model.json ({len(features)} features, {opt_rounds} rounds)")
    logger.info(f"  Train:     {TRAIN_YEARS} (all sensors)")
    logger.info(f"  Test:      {TEST_YEAR}")
    logger.info(f"  MAE:       {ma['MAE']:.3f}°C (temporal holdout)")
    logger.info(f"  Hit ±1°C:  {ma['Hit_10C']*100:.1f}%")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()