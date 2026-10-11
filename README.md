# Seoul street-level air temperature (BAE 2026)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23296486.svg)](https://doi.org/10.5281/zenodo.23296486)

Training data and model code for:

> Chang, S.-W., Yoon, S. (2026). Any coordinate, any time: city-wide prediction of street-level air temperature across Seoul. *Building and Environment*, under revision (BAE-D-26-07198).

**Data:** [doi:10.5281/zenodo.23296486](https://doi.org/10.5281/zenodo.23296486) (Zenodo). Put the three `.parquet` files in `data/`.

## Quick start

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py            # ~5 s, RMSE ≈ 1.6 °C (AWS baseline 4.5 °C)
python scripts/run_config.py            # list runs
python scripts/run_config.py loro       # run one -> results/
```

The training scripts use the GPU when XGBoost finds one. The paper's main models were trained on a CPU and the ablations on an RTX 4070; compare each ablation with `reference_gpu`.

## Data

One row per sensor-hour (KST). The columns are described in [`data/columns.md`](data/columns.md). Features are stored as float32, so rebuilt tables can differ from the paper in the last digit.

| Year | Rows | Sensors |
|---|---:|---:|
| 2023 | 3,786,015 | 1,102 |
| 2024 | 7,522,800 | 1,057 |
| 2025 | 6,741,058 | 1,112 |
| Total | 18,049,873 | 1,128 |

## Runs → tables

| Table | Run | Output |
|---|---|---|
| 5 | `random8020` | `results/Random8020_Outputs/Metrics/` |
| 6 | `loro` | `results/Discovery_Outputs/Metrics/` |
| 7, 8 | `temporal_2025` | `results/Full_Model/Temporal_Holdout/` |
| A.5 | `random8020` | `results/Random8020_Outputs/Metrics/TempStratified.csv` |

The other runs are ablations (`ablation_*`), the combined spatio-temporal holdout (`holdout_st`) and SHAP (`shap_temporal`). Confidence intervals and the revision tables come from `code/analysis/` (`pip install -r requirements-analysis.txt`); each script lists its outputs in its first line.

## Not included

- The feature-extraction pipeline, the trained models and the prediction service.
- Inputs needed by `num_coverage.py`, `num_section45.py`, `num_baseline_sensitivity.py`, table C1 and `ablation_nearest`.

## Sources

The dataset contains derived values only. Cite the original providers:

- S-DoT, Seoul Open Data Plaza
- KMA AWS/ASOS, KMA Open MET Data Portal
- NGII 1:1,000 topographic map, VWorld
- EGIS land cover, Ministry of Environment

## License

Code: MIT. Dataset: CC BY 4.0.
