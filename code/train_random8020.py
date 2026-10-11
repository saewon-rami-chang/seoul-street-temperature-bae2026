"""Random 80/20 sensor holdout, using the published test-sensor list in code/configs/."""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import gc
import json
import logging
import numpy as np
import pandas as pd
import polars as pl
import seoul_data
import xgboost as xgb
from pathlib import Path
from pyproj import Transformer
from sklearn.model_selection import GroupKFold
from scipy import stats as scipy_stats
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring",  5: "Spring",
    6: "Summer",  7: "Summer",  8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
META_COLS = {"serial_no", "datetime", "lat", "lon", "aws_temp_interp", "target_resid"}
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
OUTLIER_THRESHOLD = 25.0

TEST_FRACTION = 0.20
RANDOM_SEED = 41


def detect_device():
    """Auto-detect CUDA GPU. Returns 'cuda' or 'cpu'."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("GPU detected — using CUDA")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.info("No GPU detected — using CPU")
    return "cpu"


class PipelineConfig:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        out_dir: str | Path   = "Random8020_Outputs",
        grid_size: int         = 2500,
        n_folds: int           = 3,
        min_year: int          = 2023,
    ):
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else seoul_data.default_data_dir()
        self.out_dir = (seoul_data.default_results_dir() / out_dir).resolve()
        self.grid_size = grid_size
        self.n_folds = n_folds
        self.min_year = min_year

        device = detect_device()
        self.xgb_params = {
            "objective":        "reg:absoluteerror",
            "eval_metric":      "mae",
            "tree_method":      "hist",
            "device":           device,
            "max_bin":          256,
            "n_jobs":           6,
            "learning_rate":    0.05,
            "max_depth":        12,
            "min_child_weight": 50,
            "subsample":        0.9,
            "colsample_bytree": 0.55,
            "gamma":            0.7,
        }

    def make_dirs(self):
        dirs = {}
        for name in ["Models", "Metrics", "Predictions",
                      "Convergence", "Spatial"]:
            d = self.out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            dirs[name] = d
        return dirs


class _ChunkIter(xgb.DataIter):
    def __init__(self, files, features, serial_nos, target="target_resid"):
        self._files = files
        self._features = features
        self._sids = serial_nos
        self._target = target
        self._it = 0
        super().__init__()

    def next(self, input_data):
        while self._it < len(self._files):
            df = self._files[self._it].read()
            self._it += 1
            df = df.filter(
                pl.col("serial_no").is_in(self._sids) &
                pl.col(self._target).is_not_null() &
                (pl.col(self._target).abs() < OUTLIER_THRESHOLD)
            )
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


class DataLoader:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.features: list[str] = []
        self.files: list[Path] = []
        self.sensor_map: pd.DataFrame = pd.DataFrame()

    def build_sensor_map(self):
        all_files = seoul_data.units(self.cfg.cache_dir)
        self.files = [u for u in all_files if u.year >= self.cfg.min_year]
        logger.info(f">>> SENSOR MAP from {len(self.files)} chunks "
                     f"({self.cfg.min_year}–2025, skipped {len(all_files) - len(self.files)})...")

        schema_cols = seoul_data.schema_names(self.files)
        self.features = sorted(schema_cols - META_COLS)
        logger.info(f"    {len(self.features)} features.")

        sm = (
            seoul_data.scan(self.files)
            .select(["serial_no", "lat", "lon"])
            .unique().collect().to_pandas()
        )
        x, y = _TRANSFORMER.transform(sm["lon"].values, sm["lat"].values)
        sm["x"] = x.astype(np.float32)
        sm["y"] = y.astype(np.float32)

        self.sensor_map = sm
        logger.info(f"    {len(sm)} sensors ready.")

    def get_random_split(self, test_frac: float = TEST_FRACTION,
                         seed: int = RANDOM_SEED) -> dict[str, set]:
        """
        Standard 80/20 random sensor-level split.
        No spatial buffer — sensors are shuffled uniformly at random.
        """
        sm = self.sensor_map
        all_sids = sm["serial_no"].unique()
        # published test sensors: the seed alone does not reproduce them (polars .unique() order varies)
        split_file = Path(__file__).resolve().parent / "configs" / "random8020_test_sensors.txt"
        if split_file.exists():
            listed = {s.strip() for s in split_file.read_text().split() if s.strip()}
            holdout_sids = listed & set(all_sids)
            train_sids = set(all_sids) - holdout_sids
            logger.info(f"    Test sensors from {split_file.name} ({len(listed)} listed)")
        else:
            rng = np.random.RandomState(seed)
            rng.shuffle(all_sids)
            n_test = max(1, int(len(all_sids) * test_frac))
            holdout_sids = set(all_sids[:n_test])
            train_sids = set(all_sids[n_test:])

        logger.info(f"    Random 80/20 split (seed={seed}):")
        logger.info(f"      Train: {len(train_sids)}  Holdout: {len(holdout_sids)}  "
                     f"({len(holdout_sids)/len(all_sids)*100:.1f}% test)")
        return {"holdout": holdout_sids, "train": train_sids}

    def assign_folds(self, train_sids: set) -> dict[str, int]:
        """Spatial-grouped CV folds within the training set (same as LORO)."""
        sm = self.sensor_map
        train_sm = sm[sm["serial_no"].isin(train_sids)].copy()
        gs = self.cfg.grid_size
        train_sm["grid_id"] = (
            (train_sm["x"] // gs).astype(int).astype(str) + "_" +
            (train_sm["y"] // gs).astype(int).astype(str)
        )
        gkf = GroupKFold(n_splits=self.cfg.n_folds)
        fold_map = {}
        for fold, (_, vi) in enumerate(gkf.split(train_sm, groups=train_sm["grid_id"])):
            for sid in train_sm.iloc[vi]["serial_no"].values:
                fold_map[sid] = fold
        logger.info(f"      {train_sm['grid_id'].nunique()} grid cells -> {self.cfg.n_folds} folds")
        return fold_map

    def make_train_dmatrix(self, serial_nos: set) -> xgb.QuantileDMatrix:
        it = _ChunkIter(self.files, self.features, serial_nos)
        return xgb.QuantileDMatrix(it, max_bin=256)

    def make_val_dmatrix(self, serial_nos: set, ref) -> xgb.QuantileDMatrix:
        it = _ChunkIter(self.files, self.features, serial_nos)
        return xgb.QuantileDMatrix(it, ref=ref, max_bin=256)

    def load_holdout(self, holdout_sids: set) -> pl.DataFrame:
        df = (
            seoul_data.scan(self.files)
            .filter(
                pl.col("serial_no").is_in(list(holdout_sids)) &
                pl.col("target_resid").is_not_null() &
                (pl.col("target_resid").abs() < OUTLIER_THRESHOLD)
            )
            .with_columns([pl.col(c).cast(pl.Float32) for c in self.features])
            .collect()
        )
        x, y = _TRANSFORMER.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
        df = df.with_columns([
            pl.Series("x_coord", x, dtype=pl.Float32),
            pl.Series("y_coord", y, dtype=pl.Float32),
        ])
        return df


class TrainingEngine:
    def __init__(self, cfg: PipelineConfig, loader: DataLoader):
        self.cfg = cfg
        self.loader = loader
        self.features = loader.features
        self.dirs = cfg.make_dirs()
        self.model: xgb.Booster | None = None
        self.all_evals: list[dict] = []
        self.best_iters: list[int] = []
        self.cv_summary_rows: list[dict] = []

    @staticmethod
    def _make_cb(patience=40, min_delta=0.0001, window=25):
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

    def _train(self, split: dict, fold_map: dict):
        params = self.cfg.xgb_params.copy()
        nf = self.cfg.n_folds

        logger.info(f"    {nf}-fold CV...")
        for fold in range(nf):
            fold_train = {s for s, f in fold_map.items() if f != fold}
            fold_val   = {s for s, f in fold_map.items() if f == fold}
            logger.info(f"      Fold {fold+1}: building DMatrices...")
            dtrain = self.loader.make_train_dmatrix(fold_train)
            dval   = self.loader.make_val_dmatrix(fold_val, ref=dtrain)

            er: dict = {}
            cb = self._make_cb()
            xgb.train(params, dtrain, num_boost_round=3000,
                      evals=[(dtrain, "train"), (dval, "valid")],
                      evals_result=er, callbacks=[cb], verbose_eval=False)

            self.cv_summary_rows.append({
                "Region": "Random8020", "Fold": fold + 1,
                "Best_Round": cb.best_round, "Val_MAE": cb.best_score,
                "N_Train_Sensors": len(fold_train), "N_Val_Sensors": len(fold_val),
            })
            self.best_iters.append(cb.best_round)
            self.all_evals.append(er)
            logger.info(f"      Fold {fold+1} -> rnd={cb.best_round} MAE={cb.best_score:.5f}")
            del dtrain, dval; gc.collect()

        arr = np.array(self.best_iters)
        cv = arr.std() / arr.mean() if arr.mean() > 0 else 0
        opt = int(np.median(arr)) if cv > 0.30 else int(np.mean(arr))
        method = "median" if cv > 0.30 else "mean"
        logger.info(f"    Rounds: {list(arr)} -> {method}={opt} (CV={cv:.2f})")

        self.cv_summary_rows.append({
            "Region": "Random8020", "Fold": "Master",
            "Best_Round": opt, "Val_MAE": np.nan,
            "N_Train_Sensors": len(split["train"]), "N_Val_Sensors": 0,
        })

        logger.info(f"    Training master ({opt} rounds)...")
        dtrain_full = self.loader.make_train_dmatrix(split["train"])
        model = xgb.train(params, dtrain_full, num_boost_round=opt,
                          evals=[(dtrain_full, "train")], verbose_eval=100)
        del dtrain_full; gc.collect()

        model.feature_names = self.features
        self.model = model

        model.save_model(str(self.dirs["Models"] / "Model_Random8020.json"))
        gain = model.get_score(importance_type="gain")
        wt = model.get_score(importance_type="weight")
        pd.DataFrame({
            "Feature": list(gain.keys()), "Gain": list(gain.values()),
            "Weight": [wt.get(k, 0) for k in gain.keys()],
        }).sort_values("Gain", ascending=False).to_csv(
            self.dirs["Models"] / "Native_Importance_Random8020.csv", index=False)
        logger.info("    Saved Models/Model_Random8020.json")

    def train_and_evaluate(self) -> pd.DataFrame:
        logger.info(f"\n{'='*70}")
        logger.info(f"  RANDOM 80/20 SENSOR-LEVEL SPLIT")
        logger.info(f"  Seed={RANDOM_SEED}  Test={TEST_FRACTION*100:.0f}%")
        logger.info(f"{'='*70}")

        split = self.loader.get_random_split()

        model_path = self.dirs["Models"] / "Model_Random8020.json"
        if model_path.exists():
            logger.info(f"    [FOUND — skipping training]")
            model = xgb.Booster()
            model.load_model(str(model_path))
            model.feature_names = self.features
            self.model = model
            logger.info(f"    Loaded {model_path.name}")
        else:
            fold_map = self.loader.assign_folds(split["train"])
            self._train(split, fold_map)

        logger.info("    Loading holdout data...")
        holdout_df = self.loader.load_holdout(split["holdout"])
        model = self.model
        X_h = holdout_df.select(self.features).to_numpy().astype(np.float32)
        dm = xgb.DMatrix(X_h, feature_names=self.features)
        pred_resid = model.predict(dm)
        del dm, X_h; gc.collect()

        aws = holdout_df["aws_temp_interp"].to_numpy()
        tgt = holdout_df["target_resid"].to_numpy()
        frame = holdout_df.select(
            ["datetime", "serial_no", "lat", "lon", "x_coord", "y_coord",
             "aws_temp_interp", "target_resid"]
        ).to_pandas()
        frame["pred_resid"]  = pred_resid
        frame["actual_temp"] = (aws + tgt).astype(np.float32)
        frame["pred_temp"]   = (aws + pred_resid).astype(np.float32)
        frame["region"]      = "Random8020"
        del holdout_df; gc.collect()

        frame["datetime"]   = pd.to_datetime(frame["datetime"])
        frame["Error"]      = frame["pred_temp"] - frame["actual_temp"]
        frame["eval_hour"]  = frame["datetime"].dt.hour.astype(int)
        frame["eval_month"] = frame["datetime"].dt.month.astype(int)
        frame["season"]     = frame["eval_month"].map(MONTH_TO_SEASON)
        return frame

    @staticmethod
    def _compute_metrics(y_true, y_pred, full=True) -> dict:
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

    @staticmethod
    def _box_stats(df, col):
        return (
            df.groupby(col)["Error"]
            .agg(Mean="mean", Median="median", Std="std",
                 Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75), N="count")
            .reset_index()
        )

    def save_all_data(self, df_eval: pd.DataFrame):
        d = self.dirs
        sm = self.loader.sensor_map
        logger.info(f"\n{'='*70}\nSAVING ALL DATA\n{'='*70}")

        logger.info(">>> Predictions...")
        df_eval.to_parquet(d["Predictions"] / "Predictions.parquet", index=False)

        sm.to_csv(d["Spatial"] / "Sensor_Map.csv", index=False)

        logger.info(">>> Convergence...")
        pd.DataFrame(self.cv_summary_rows).to_csv(
            d["Convergence"] / "CV_Fold_Summary.csv", index=False)
        conv = []
        for fold, er in enumerate(self.all_evals):
            for sp in ("train", "valid"):
                if sp not in er:
                    continue
                for rnd, val in enumerate(er[sp]["mae"]):
                    conv.append({"Region": "Random8020", "Fold": fold + 1,
                                 "Round": rnd, "Split": sp, "MAE": val})
        pd.DataFrame(conv).to_csv(d["Convergence"] / "Convergence_Data.csv", index=False)

        logger.info(">>> Metrics...")
        ns = df_eval["serial_no"].nunique()

        ma = self._compute_metrics(df_eval["actual_temp"], df_eval["pred_temp"], full=True)
        ba = self._compute_metrics(df_eval["actual_temp"], df_eval["aws_temp_interp"], full=True)
        mr = self._compute_metrics(df_eval["target_resid"], df_eval["pred_resid"], full=False)

        rows_abs = [
            {"Region": "Random8020", "Model": "XGBoost", "N_sensors": ns, **ma},
            {"Region": "Random8020", "Model": "AWS Baseline", "N_sensors": ns, **ba},
        ]
        rows_res = [{"Region": "Random8020", "N_sensors": ns, **mr}]

        logger.info(f"  {'Random8020':35s} ({ns:3d} sens) abs MAE={ma['MAE']:.3f} R2={ma['R2']:.4f} | "
                     f"res MAE={mr['MAE']:.3f} R2={mr['R2']:.4f}")

        da = pd.DataFrame(rows_abs); dr = pd.DataFrame(rows_res)
        da.to_csv(d["Metrics"] / "Global_Metrics_Absolute.csv", index=False)
        dr.to_csv(d["Metrics"] / "Global_Metrics_Residual.csv", index=False)

        m = da[da["Model"] == "XGBoost"].iloc[0]
        b = da[da["Model"] == "AWS Baseline"].iloc[0]
        imp = [{"Region": "Random8020",
            "XGB_MAE": m["MAE"], "AWS_MAE": b["MAE"],
            "Delta_MAE": m["MAE"] - b["MAE"],
            "Pct_MAE": (b["MAE"] - m["MAE"]) / b["MAE"] * 100,
            "XGB_RMSE": m["RMSE"], "AWS_RMSE": b["RMSE"],
            "Delta_RMSE": m["RMSE"] - b["RMSE"],
            "Pct_RMSE": (b["RMSE"] - m["RMSE"]) / b["RMSE"] * 100}]
        pd.DataFrame(imp).to_csv(d["Metrics"] / "Baseline_Improvement.csv", index=False)

        logger.info(">>> Per-sensor...")
        srows = []
        for sno, g in df_eval.groupby("serial_no"):
            if len(g) < 24: continue
            ma_s = self._compute_metrics(g["actual_temp"], g["pred_temp"], full=False)
            mr_s = self._compute_metrics(g["target_resid"], g["pred_resid"], full=False)
            srows.append({
                "serial_no": sno, "region": "Random8020",
                "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0],
                "x_coord": g["x_coord"].iloc[0], "y_coord": g["y_coord"].iloc[0],
                "N_hours": len(g),
                "abs_MAE": ma_s["MAE"], "abs_RMSE": ma_s["RMSE"],
                "abs_MBE": ma_s["MBE"], "abs_R2": ma_s["R2"],
                "res_MAE": mr_s["MAE"], "res_RMSE": mr_s["RMSE"],
                "res_MBE": mr_s["MBE"], "res_R2": mr_s["R2"]})
        sensor_df = pd.DataFrame(srows)
        sensor_df.to_csv(d["Metrics"] / "PerSensor_Performance.csv", index=False)

        summ = []
        for col in ["abs_MAE", "abs_RMSE", "abs_MBE", "res_MAE"]:
            summ.append({"Region": "Random8020", "Metric": col, "N": len(sensor_df),
                "Mean": sensor_df[col].mean(), "Std": sensor_df[col].std(),
                "Median": sensor_df[col].median(),
                "Min": sensor_df[col].min(), "Max": sensor_df[col].max()})
        pd.DataFrame(summ).to_csv(d["Metrics"] / "PerSensor_Summary.csv", index=False)
        logger.info(f"    {len(sensor_df)} sensors")

        logger.info(">>> Temporal stats...")
        self._box_stats(df_eval, "eval_hour").rename(
            columns={"eval_hour": "Hour"}).to_csv(d["Metrics"] / "Diurnal_Stats.csv", index=False)
        self._box_stats(df_eval, "eval_month").rename(
            columns={"eval_month": "Month"}).to_csv(d["Metrics"] / "Monthly_Stats.csv", index=False)
        sd = []
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            sub = df_eval[df_eval["season"] == season]
            if len(sub) == 0: continue
            st = sub.groupby("eval_hour")["Error"].agg(
                Mean="mean", Std="std", N="count",
                Q25=lambda x: x.quantile(0.25), Q75=lambda x: x.quantile(0.75)).reset_index()
            st.insert(0, "Season", season); sd.append(st)
        if sd:
            pd.concat(sd, ignore_index=True).to_csv(d["Metrics"] / "SeasonDiurnal_Stats.csv", index=False)

        bins = [-999, 0, 10, 20, 30, 999]
        labels = ["<0C", "0-10C", "10-20C", "20-30C", ">30C"]
        df_eval["temp_bin"] = pd.cut(df_eval["actual_temp"], bins=bins, labels=labels)
        tstrat = []
        for bl in labels:
            sub = df_eval[df_eval["temp_bin"] == bl]
            if len(sub) < 10: continue
            tstrat.append({"TempBin": bl, "Model": "XGBoost",
                           **self._compute_metrics(sub["actual_temp"], sub["pred_temp"])})
            tstrat.append({"TempBin": bl, "Model": "AWS Baseline",
                           **self._compute_metrics(sub["actual_temp"], sub["aws_temp_interp"])})
        pd.DataFrame(tstrat).to_csv(d["Metrics"] / "TempStratified.csv", index=False)

        pd.DataFrame([{
            "MBE": float(df_eval["Error"].mean()), "Std": float(df_eval["Error"].std()),
            "Skew": float(scipy_stats.skew(df_eval["Error"])),
            "Kurt": float(scipy_stats.kurtosis(df_eval["Error"])),
            "Min": float(df_eval["Error"].min()), "Max": float(df_eval["Error"].max()),
            "Hit_05C": float((df_eval["Error"].abs() <= 0.5).mean()),
            "Hit_10C": float((df_eval["Error"].abs() <= 1.0).mean()),
            "Hit_20C": float((df_eval["Error"].abs() <= 2.0).mean()),
        }]).to_csv(d["Metrics"] / "Error_Distribution.csv", index=False)

        logger.info(f"\n{'='*70}\nRANDOM 80/20 COMPLETE — {self.cfg.out_dir}/\n{'='*70}")


if __name__ == "__main__":
    cfg = PipelineConfig()
    loader = DataLoader(cfg)
    loader.build_sensor_map()

    engine = TrainingEngine(cfg, loader)
    df_eval = engine.train_and_evaluate()
    engine.save_all_data(df_eval)

    del df_eval; gc.collect()