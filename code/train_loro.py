"""Leave-one-region-out validation (6 regions, 1 km buffers). Flags select the ablations (see code/configs/)."""

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
BUFFER_M = 1000
OUTLIER_THRESHOLD = 25.0

RESOLUTIONS = ("10m", "100m", "250m", "1000m")
LAG_FEATURES = {  # 25 rolling/lag features, Table A.4
    "aws_temp_lag_1h", "aws_temp_lag_2h", "aws_temp_lag_3h", "aws_temp_lag_6h",
    "aws_temp_mean_6h", "aws_temp_mean_24h",
    "aws_temp_trend_1h", "aws_temp_trend_3h", "aws_temp_trend_24h", "daily_temp_swing",
    "wind_u_mean_3h", "wind_u_mean_6h", "wind_v_mean_3h", "wind_v_mean_6h",
    "solar_sum_6h", "solar_sum_12h", "cloud_mean_3h", "cloud_mean_24h",
    "precip_sum_3h", "precip_sum_24h", "humidity_mean_3h", "humidity_mean_12h",
    "dew_point_mean_3h", "dew_point_mean_12h", "pressure_trend_3h",
}


def features_in_group(group: str, features: list[str]) -> set[str]:
    """Features removed by `--exclude <group>`."""
    if group == "spatial":
        return {f for f in features if f.endswith(tuple(f"_{r}" for r in RESOLUTIONS))}
    if group == "shadow":
        return {f for f in features if f.startswith("shadow_fraction_")}
    if group == "svf":
        return {f for f in features if f.startswith("SVF_")}
    if group == "lag":
        return LAG_FEATURES & set(features)
    if group.startswith("scale") and group[5:] + "m" in RESOLUTIONS:
        return {f for f in features if f.endswith(f"_{group[5:]}m")}
    raise ValueError(f"Unknown exclude group: {group}")

REGIONS = {
    "A_Mapo": {
        "label": "A: Mapo / Sangam DMC",
        "desc":  "Mixed composite — low-rise, high-rise DMC, Han River, World Cup Park",
        "lat_min": 37.535, "lat_max": 37.580, "lon_min": 126.880, "lon_max": 126.930,
    },
    "B_Gangnam": {
        "label": "B: Gangnam / Seocho CBD",
        "desc":  "Dense high-rise commercial — towers, wide boulevards, high impervious",
        "lat_min": 37.475, "lat_max": 37.520, "lon_min": 127.000, "lon_max": 127.075,
    },
    "C_Central": {
        "label": "C: Central Seoul",
        "desc":  "Historic/government core — Gwanghwamun, Seoul Station, Gyeongbokgung",
        "lat_min": 37.555, "lat_max": 37.585, "lon_min": 126.965, "lon_max": 127.025,
    },
    "D_Gangdong": {
        "label": "D: Gangdong (East)",
        "desc":  "Eastern residential — mid/high-rise apartments, suburban fabric",
        "lat_min": 37.530, "lat_max": 37.565, "lon_min": 127.115, "lon_max": 127.183,
    },
    "E_Dobong": {
        "label": "E: Dobong / Nowon (North)",
        "desc":  "Northern less-affluent residential — low/mid-rise, Bukhansan foothills",
        "lat_min": 37.640, "lat_max": 37.675, "lon_min": 127.020, "lon_max": 127.065,
    },
    "F_Gwanak": {
        "label": "F: Gwanak / Sillim",
        "desc":  "Mountain-influence villa district — Gwanaksan cooling, steep terrain edge",
        "lat_min": 37.455, "lat_max": 37.490, "lon_min": 126.915, "lon_max": 126.960,
    },
}


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
        out_dir: str | Path   = "Discovery_Outputs",
        grid_size: int         = 2500,
        n_folds: int           = 3,
        min_year: int          = 2023,
        exclude: list[str] | None = None,
        target: str            = "residual",
        holdout_year: int | None = None,
        sensors: str = "all",
    ):
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else seoul_data.default_data_dir()
        self.out_dir = (seoul_data.default_results_dir() / out_dir).resolve()
        self.grid_size = grid_size
        self.n_folds = n_folds
        self.min_year = min_year
        self.exclude = exclude or []
        self.target = target              # "residual" | "absolute"
        self.holdout_year = holdout_year  # None = LORO over all years
        self.sensors = sensors            # "all" | "v02q" (serial prefix V02Q only)

        device = detect_device()
        self.xgb_params = {
            "objective":        "reg:absoluteerror",
            "eval_metric":      "mae",
            "tree_method":      "hist",
            "device":           device,
            "max_bin":          256,
            "n_jobs":           8,
            "learning_rate":    0.05,
            "max_depth":        12,
            "min_child_weight": 50,
            "subsample":        0.9,
            "colsample_bytree": 0.55,
            "gamma":            0.7,
        }

    def make_dirs(self):
        """Create organized output subdirectories."""
        dirs = {}
        for name in ["Models", "Metrics", "Predictions",
                      "Convergence", "Spatial"]:
            d = self.out_dir / name
            d.mkdir(parents=True, exist_ok=True)
            dirs[name] = d
        return dirs


class _ChunkIter(xgb.DataIter):
    def __init__(self, files, features, serial_nos, target="residual"):
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
            # Row filter is always on the residual so every run sees the same rows
            df = df.filter(
                pl.col("serial_no").is_in(self._sids) &
                pl.col("target_resid").is_not_null() &
                (pl.col("target_resid").abs() < OUTLIER_THRESHOLD)
            )
            if df.height > 0:
                X = df.select(self._features).to_numpy().astype(np.float32)
                y = df["target_resid"].to_numpy().astype(np.float32)
                if self._target == "absolute":
                    y = y + df["aws_temp_interp"].to_numpy().astype(np.float32)
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
        self.excluded: list[str] = []
        self.files: list[Path] = []
        self.train_files: list[Path] = []
        self.test_files: list[Path] = []
        self.sensor_map: pd.DataFrame = pd.DataFrame()
        self.region_bounds_5179: dict[str, dict] = {}

    def build_sensor_map(self):
        all_files = seoul_data.units(self.cfg.cache_dir)
        self.files = [u for u in all_files if u.year >= self.cfg.min_year]
        logger.info(f">>> SENSOR MAP from {len(self.files)} chunks "
                     f"({self.cfg.min_year}–2025, skipped {len(all_files) - len(self.files)})...")

        # Train/test files: same set for plain LORO; split by year for --holdout-years
        hy = self.cfg.holdout_year
        year = lambda u: u.year
        if hy is None:
            self.train_files = self.test_files = self.files
        else:
            self.train_files = [f for f in self.files if year(f) < hy]
            self.test_files = [f for f in self.files if year(f) == hy]
            logger.info(f"    Holdout year {hy}: {len(self.train_files)} train chunks, "
                         f"{len(self.test_files)} test chunks")

        schema_cols = seoul_data.schema_names(self.files)
        all_features = sorted(schema_cols - META_COLS)
        drop = set()
        for g in self.cfg.exclude:
            drop |= features_in_group(g, all_features)
        self.excluded = sorted(drop)
        self.features = [f for f in all_features if f not in drop]
        logger.info(f"    {len(self.features)} features "
                     f"({len(self.excluded)} excluded: {self.cfg.exclude or 'none'}).")

        sm = (
            seoul_data.scan(self.files)
            .select(["serial_no", "lat", "lon"])
            .unique().collect().to_pandas()
        )
        x, y = _TRANSFORMER.transform(sm["lon"].values, sm["lat"].values)
        sm["x"] = x.astype(np.float32)
        sm["y"] = y.astype(np.float32)

        for key, r in REGIONS.items():
            xn, yn = _TRANSFORMER.transform(r["lon_min"], r["lat_min"])
            xx, yx = _TRANSFORMER.transform(r["lon_max"], r["lat_max"])
            self.region_bounds_5179[key] = {"x_min": xn, "x_max": xx, "y_min": yn, "y_max": yx}

        if self.cfg.sensors == "v02q":
            n0 = len(sm)
            sm = sm[sm["serial_no"].str.startswith("V02Q")].reset_index(drop=True)
            logger.info(f"    Sensor subset V02Q: {len(sm)} of {n0} sensors kept.")
        self.sensor_map = sm
        logger.info(f"    {len(sm)} sensors ready.")

    def get_region_split(self, region_key: str) -> dict[str, set]:
        sm = self.sensor_map
        b = self.region_bounds_5179[region_key]
        in_box = (
            (sm["x"] >= b["x_min"]) & (sm["x"] <= b["x_max"]) &
            (sm["y"] >= b["y_min"]) & (sm["y"] <= b["y_max"])
        )
        in_buffer = (
            (sm["x"] >= b["x_min"] - BUFFER_M) & (sm["x"] <= b["x_max"] + BUFFER_M) &
            (sm["y"] >= b["y_min"] - BUFFER_M) & (sm["y"] <= b["y_max"] + BUFFER_M) &
            ~in_box
        )
        holdout_sids = set(sm.loc[in_box, "serial_no"])
        buffer_sids  = set(sm.loc[in_buffer, "serial_no"])
        train_sids   = set(sm["serial_no"]) - holdout_sids - buffer_sids
        logger.info(f"      Holdout: {len(holdout_sids)}  Buffer: {len(buffer_sids)}  Train: {len(train_sids)}")
        return {"holdout": holdout_sids, "buffer": buffer_sids, "train": train_sids}

    def assign_folds(self, train_sids: set) -> dict[str, int]:
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
        it = _ChunkIter(self.train_files, self.features, serial_nos, self.cfg.target)
        return xgb.QuantileDMatrix(it, max_bin=256)

    def make_val_dmatrix(self, serial_nos: set, ref) -> xgb.QuantileDMatrix:
        it = _ChunkIter(self.train_files, self.features, serial_nos, self.cfg.target)
        return xgb.QuantileDMatrix(it, ref=ref, max_bin=256)

    def load_holdout(self, holdout_sids: set) -> pl.DataFrame:
        df = (
            seoul_data.scan(self.test_files)
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
        self.region_bounds_5179 = loader.region_bounds_5179
        self.dirs = cfg.make_dirs()
        self.models: dict[str, xgb.Booster] = {}
        self.all_evals: dict[str, list[dict]] = {}
        self.all_best_iters: dict[str, list[int]] = {}
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

    def _train_one(self, region_key: str, split: dict, fold_map: dict):
        params = self.cfg.xgb_params.copy()
        nf = self.cfg.n_folds
        evals_list, best_iters = [], []

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
                "Region": region_key, "Fold": fold + 1,
                "Best_Round": cb.best_round, "Val_MAE": cb.best_score,
                "N_Train_Sensors": len(fold_train), "N_Val_Sensors": len(fold_val),
            })
            best_iters.append(cb.best_round)
            evals_list.append(er)
            logger.info(f"      Fold {fold+1} -> rnd={cb.best_round} MAE={cb.best_score:.5f}")
            del dtrain, dval; gc.collect()

        arr = np.array(best_iters)
        cv = arr.std() / arr.mean() if arr.mean() > 0 else 0
        opt = int(np.median(arr)) if cv > 0.30 else int(np.mean(arr))
        method = "median" if cv > 0.30 else "mean"
        logger.info(f"    Rounds: {list(arr)} -> {method}={opt} (CV={cv:.2f})")

        self.cv_summary_rows.append({
            "Region": region_key, "Fold": "Master",
            "Best_Round": opt, "Val_MAE": np.nan,
            "N_Train_Sensors": len(split["train"]), "N_Val_Sensors": 0,
        })

        logger.info(f"    Training master ({opt} rounds)...")
        dtrain_full = self.loader.make_train_dmatrix(split["train"])
        model = xgb.train(params, dtrain_full, num_boost_round=opt,
                          evals=[(dtrain_full, "train")], verbose_eval=100)
        del dtrain_full; gc.collect()

        model.feature_names = self.features
        self.models[region_key] = model
        self.all_evals[region_key] = evals_list
        self.all_best_iters[region_key] = best_iters

        model.save_model(str(self.dirs["Models"] / f"Model_{region_key}.json"))
        gain = model.get_score(importance_type="gain")
        wt = model.get_score(importance_type="weight")
        pd.DataFrame({
            "Feature": list(gain.keys()), "Gain": list(gain.values()),
            "Weight": [wt.get(k, 0) for k in gain.keys()],
        }).sort_values("Gain", ascending=False).to_csv(
            self.dirs["Models"] / f"Native_Importance_{region_key}.csv", index=False)
        logger.info(f"    Saved Models/Model_{region_key}.json")

    def train_all(self, region_keys=None) -> pd.DataFrame:
        eval_frames = []
        regions_to_run = {k: REGIONS[k] for k in (region_keys or REGIONS.keys())}
        n_total = len(regions_to_run)

        model_dir = self.dirs["Models"]
        done_count = 0

        for i, key in enumerate(regions_to_run):
            model_path = model_dir / f"Model_{key}.json"
            found = model_path.exists()

            logger.info(f"\n{'='*70}")
            tag = " [FOUND — skipping training]" if found else ""
            logger.info(f"  REGION {i+1}/{n_total}: {REGIONS[key]['label']}{tag}")
            logger.info(f"  {REGIONS[key]['desc']}")
            logger.info(f"{'='*70}")

            split = self.loader.get_region_split(key)

            if found:
                model = xgb.Booster()
                model.load_model(str(model_path))
                model.feature_names = self.features
                self.models[key] = model
                self.all_evals[key] = []
                self.all_best_iters[key] = []
                logger.info(f"    Loaded {model_path.name}")
            else:
                fold_map = self.loader.assign_folds(split["train"])
                self._train_one(key, split, fold_map)

            logger.info("    Loading holdout data...")
            holdout_df = self.loader.load_holdout(split["holdout"])
            model = self.models[key]
            X_h = holdout_df.select(self.features).to_numpy().astype(np.float32)
            dm = xgb.DMatrix(X_h, feature_names=self.features)
            pred = model.predict(dm)
            del dm, X_h; gc.collect()

            aws = holdout_df["aws_temp_interp"].to_numpy()
            # absolute-target runs: convert predictions to residuals
            pred_resid = pred - aws if self.cfg.target == "absolute" else pred
            tgt = holdout_df["target_resid"].to_numpy()
            frame = holdout_df.select(
                ["datetime", "serial_no", "lat", "lon", "x_coord", "y_coord",
                 "aws_temp_interp", "target_resid"]
            ).to_pandas()
            frame["pred_resid"]  = pred_resid
            frame["actual_temp"] = (aws + tgt).astype(np.float32)
            frame["pred_temp"]   = (aws + pred_resid).astype(np.float32)
            frame["region"]      = key
            eval_frames.append(frame)
            del holdout_df; gc.collect()

            done_count += 1
            self._save_incremental(eval_frames, key, done_count - 1)

        df_eval = pd.concat(eval_frames, ignore_index=True)
        del eval_frames
        df_eval["datetime"]   = pd.to_datetime(df_eval["datetime"])
        df_eval["Error"]      = df_eval["pred_temp"] - df_eval["actual_temp"]
        df_eval["eval_hour"]  = df_eval["datetime"].dt.hour.astype(int)
        df_eval["eval_month"] = df_eval["datetime"].dt.month.astype(int)
        df_eval["season"]     = df_eval["eval_month"].map(MONTH_TO_SEASON)
        return df_eval

    def _save_incremental(self, eval_frames: list, region_key: str, region_idx: int):
        """Save partial results after each region completes."""
        d = self.dirs
        n_done = region_idx + 1

        df_partial = pd.concat(eval_frames, ignore_index=True)
        df_partial["datetime"] = pd.to_datetime(df_partial["datetime"])
        df_partial["Error"] = df_partial["pred_temp"] - df_partial["actual_temp"]

        done_keys = df_partial["region"].unique().tolist()
        logger.info(f"    Saving incremental results ({n_done} regions)...")
        rows_abs, rows_res = [], []
        for key in done_keys:
            sub = df_partial[df_partial["region"] == key]
            if len(sub) == 0: continue
            label = REGIONS[key]["label"]
            ns = sub["serial_no"].nunique()
            ma = self._compute_metrics(sub["actual_temp"], sub["pred_temp"], full=True)
            ba = self._compute_metrics(sub["actual_temp"], sub["aws_temp_interp"], full=True)
            rows_abs.append({"Region": label, "Model": "XGBoost", "N_sensors": ns, **ma})
            rows_abs.append({"Region": label, "Model": "AWS Baseline", "N_sensors": ns, **ba})
            mr = self._compute_metrics(sub["target_resid"], sub["pred_resid"], full=False)
            rows_res.append({"Region": label, "N_sensors": ns, **mr})
            logger.info(f"    {label:35s} ({ns:3d} sens) "
                         f"abs MAE={ma['MAE']:.3f} R²={ma['R2']:.4f} | "
                         f"res MAE={mr['MAE']:.3f} R²={mr['R2']:.4f}")

        ns_p = df_partial["serial_no"].nunique()
        ma_p = self._compute_metrics(df_partial["actual_temp"], df_partial["pred_temp"], full=True)
        ba_p = self._compute_metrics(df_partial["actual_temp"], df_partial["aws_temp_interp"], full=True)
        mr_p = self._compute_metrics(df_partial["target_resid"], df_partial["pred_resid"], full=False)
        rows_abs.append({"Region": f"Pooled ({n_done})", "Model": "XGBoost", "N_sensors": ns_p, **ma_p})
        rows_abs.append({"Region": f"Pooled ({n_done})", "Model": "AWS Baseline", "N_sensors": ns_p, **ba_p})
        rows_res.append({"Region": f"Pooled ({n_done})", "N_sensors": ns_p, **mr_p})
        logger.info(f"    {'Pooled ('+str(n_done)+')':35s} ({ns_p:3d} sens) "
                     f"abs MAE={ma_p['MAE']:.3f} R²={ma_p['R2']:.4f} | "
                     f"res MAE={mr_p['MAE']:.3f} R²={mr_p['R2']:.4f}")

        pd.DataFrame(rows_abs).to_csv(d["Metrics"] / "Global_Metrics_Absolute.csv", index=False)
        pd.DataFrame(rows_res).to_csv(d["Metrics"] / "Global_Metrics_Residual.csv", index=False)

        pd.DataFrame(self.cv_summary_rows).to_csv(
            d["Convergence"] / "CV_Fold_Summary.csv", index=False)

        conv = []
        for k in done_keys:
            if k not in self.all_evals: continue
            for fold, er in enumerate(self.all_evals[k]):
                for sp in ("train", "valid"):
                    for rnd, val in enumerate(er[sp]["mae"]):
                        conv.append({"Region": k, "Fold": fold + 1,
                                     "Round": rnd, "Split": sp, "MAE": val})
        pd.DataFrame(conv).to_csv(d["Convergence"] / "Convergence_Data.csv", index=False)

        df_partial.to_parquet(d["Predictions"] / "Predictions.parquet", index=False)

        del df_partial

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
        with open(d["Spatial"] / "Region_Bounds_5179.json", "w") as f:
            json.dump(self.region_bounds_5179, f, indent=2)

        logger.info(">>> Convergence...")
        pd.DataFrame(self.cv_summary_rows).to_csv(
            d["Convergence"] / "CV_Fold_Summary.csv", index=False)
        conv = []
        for key in self.all_evals:
            for fold, er in enumerate(self.all_evals[key]):
                for sp in ("train", "valid"):
                    for rnd, val in enumerate(er[sp]["mae"]):
                        conv.append({"Region": key, "Fold": fold + 1,
                                     "Round": rnd, "Split": sp, "MAE": val})
        pd.DataFrame(conv).to_csv(d["Convergence"] / "Convergence_Data.csv", index=False)

        logger.info(">>> Metrics...")
        run_keys = df_eval["region"].unique().tolist()
        subsets = run_keys + ["Pooled"]
        rows_abs, rows_res = [], []
        for s in subsets:
            sub = df_eval if s == "Pooled" else df_eval[df_eval["region"] == s]
            if len(sub) == 0: continue
            label = "Pooled" if s == "Pooled" else REGIONS[s]["label"]
            ns = sub["serial_no"].nunique()
            ma = self._compute_metrics(sub["actual_temp"], sub["pred_temp"], full=True)
            ba = self._compute_metrics(sub["actual_temp"], sub["aws_temp_interp"], full=True)
            rows_abs.append({"Region": label, "Model": "XGBoost", "N_sensors": ns, **ma})
            rows_abs.append({"Region": label, "Model": "AWS Baseline", "N_sensors": ns, **ba})
            mr = self._compute_metrics(sub["target_resid"], sub["pred_resid"], full=False)
            rows_res.append({"Region": label, "N_sensors": ns, **mr})
            logger.info(f"  {label:35s} ({ns:3d} sens) abs MAE={ma['MAE']:.3f} R2={ma['R2']:.4f} | "
                         f"res MAE={mr['MAE']:.3f} R2={mr['R2']:.4f}")
        da = pd.DataFrame(rows_abs); dr = pd.DataFrame(rows_res)
        da.to_csv(d["Metrics"] / "Global_Metrics_Absolute.csv", index=False)
        dr.to_csv(d["Metrics"] / "Global_Metrics_Residual.csv", index=False)
        da[da["Model"] == "XGBoost"].to_csv(d["Metrics"] / "Cross_Region_Comparison.csv", index=False)

        imp = []
        for s in subsets:
            label = "Pooled" if s == "Pooled" else REGIONS[s]["label"]
            m = da[(da["Region"] == label) & (da["Model"] == "XGBoost")].iloc[0]
            b = da[(da["Region"] == label) & (da["Model"] == "AWS Baseline")].iloc[0]
            imp.append({"Region": label,
                "XGB_MAE": m["MAE"], "AWS_MAE": b["MAE"],
                "Delta_MAE": m["MAE"] - b["MAE"],
                "Pct_MAE": (b["MAE"] - m["MAE"]) / b["MAE"] * 100,
                "XGB_RMSE": m["RMSE"], "AWS_RMSE": b["RMSE"],
                "Delta_RMSE": m["RMSE"] - b["RMSE"],
                "Pct_RMSE": (b["RMSE"] - m["RMSE"]) / b["RMSE"] * 100})
        pd.DataFrame(imp).to_csv(d["Metrics"] / "Baseline_Improvement.csv", index=False)

        logger.info(">>> Per-sensor...")
        srows = []
        for sno, g in df_eval.groupby("serial_no"):
            if len(g) < 24: continue
            ma = self._compute_metrics(g["actual_temp"], g["pred_temp"], full=False)
            mr = self._compute_metrics(g["target_resid"], g["pred_resid"], full=False)
            srows.append({
                "serial_no": sno, "region": g["region"].iloc[0],
                "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0],
                "x_coord": g["x_coord"].iloc[0], "y_coord": g["y_coord"].iloc[0],
                "N_hours": len(g),
                "abs_MAE": ma["MAE"], "abs_RMSE": ma["RMSE"],
                "abs_MBE": ma["MBE"], "abs_R2": ma["R2"],
                "res_MAE": mr["MAE"], "res_RMSE": mr["RMSE"],
                "res_MBE": mr["MBE"], "res_R2": mr["R2"]})
        sensor_df = pd.DataFrame(srows)
        sensor_df.to_csv(d["Metrics"] / "PerSensor_Performance.csv", index=False)

        summ = []
        for s in subsets:
            label = "Pooled" if s == "Pooled" else REGIONS[s]["label"]
            sub = sensor_df if s == "Pooled" else sensor_df[sensor_df["region"] == s]
            if len(sub) == 0: continue
            for col in ["abs_MAE", "abs_RMSE", "abs_MBE", "res_MAE"]:
                summ.append({"Region": label, "Metric": col, "N": len(sub),
                    "Mean": sub[col].mean(), "Std": sub[col].std(),
                    "Median": sub[col].median(), "Min": sub[col].min(), "Max": sub[col].max()})
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

        logger.info(f"\n{'='*70}\nDISCOVERY COMPLETE — {self.cfg.out_dir}/\n{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default=None,
                        help="Comma-separated region keys to run (e.g. E_Dobong,F_Gwanak)")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated feature groups to drop: spatial, shadow, svf, "
                             "lag, scale10, scale100, scale250, scale1000")
    parser.add_argument("--target", choices=["residual", "absolute"], default="residual")
    parser.add_argument("--holdout-years", type=int, default=None,
                        help="Combined holdout: train on years before this one outside the "
                             "holdout region, test on this year inside it (e.g. 2025)")
    parser.add_argument("--sensors", choices=["all", "v02q"], default="all",
                        help="Sensor subset for training and evaluation (v02q = serial prefix V02Q). "
                             "Requires --out-dir when not 'all'.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Feature table folder (default: data/ or $SEOUL_DATA_DIR). Requires --out-dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output folder (default: Discovery_Outputs for the main run, "
                             "otherwise derived from the flags)")
    args = parser.parse_args()

    exclude = [g.strip() for g in args.exclude.split(",")] if args.exclude else []
    is_main = (not exclude and args.target == "residual" and args.holdout_years is None
               and args.cache_dir is None and args.sensors == "all")
    if (args.cache_dir or args.sensors != "all") and not args.out_dir:
        print("ERROR: --cache-dir / --sensors need an explicit --out-dir.")
        exit(1)
    if args.out_dir:
        out_dir = args.out_dir
    elif is_main:
        out_dir = "Discovery_Outputs"
    elif args.holdout_years is not None:
        out_dir = "Holdout_ST" + (f"_{'_'.join(exclude)}" if exclude else "") \
                  + ("_absolute" if args.target == "absolute" else "")
    else:
        out_dir = "Ablation_" + "_".join(exclude + (["absolute"] if args.target == "absolute" else []))
    if not is_main and Path(out_dir).name == "Discovery_Outputs":
        print("ERROR: flagged runs must not write to Discovery_Outputs (main-model outputs).")
        exit(1)

    cfg = PipelineConfig(out_dir=out_dir, exclude=exclude,
                         target=args.target, holdout_year=args.holdout_years,
                         sensors=args.sensors,
                         **({"cache_dir": args.cache_dir} if args.cache_dir else {}))
    loader = DataLoader(cfg)
    loader.build_sensor_map()
    for g in exclude:
        if not features_in_group(g, loader.features + loader.excluded):
            print(f"ERROR: --exclude {g} matched no features.")
            exit(1)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_dir / "run_config.json", "w") as f:
        json.dump({
            "cache_dir": str(cfg.cache_dir), "sensors": cfg.sensors, "exclude": exclude, "target": args.target,
            "holdout_year": args.holdout_years, "min_year": cfg.min_year,
            "n_features": len(loader.features), "excluded_features": loader.excluded,
            "xgb_params": cfg.xgb_params, "xgboost_version": xgb.__version__,
            "buffer_m": BUFFER_M, "n_folds": cfg.n_folds, "grid_size": cfg.grid_size,
        }, f, indent=2)

    if args.regions:
        run_regions = [r.strip() for r in args.regions.split(",")]
        invalid = [r for r in run_regions if r not in REGIONS]
        if invalid:
            print(f"ERROR: Unknown regions: {invalid}")
            print(f"Valid: {list(REGIONS.keys())}")
            exit(1)
    else:
        run_regions = None  # all

    engine = TrainingEngine(cfg, loader)
    df_eval = engine.train_all(region_keys=run_regions)
    engine.save_all_data(df_eval)

    del df_eval; gc.collect()