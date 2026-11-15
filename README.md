# SC-SP: Source-Calibrated Selective Prognostics for Maritime Predictive Maintenance

## Paper

*Source-Calibrated Selective Prognostics: A Deployment-Time Reliability Layer for Remaining Useful Life Prediction under Distribution Shift* (submitted to *Reliability Engineering & System Safety*).

This work follows the previous study, whose code is available at https://github.com/dongwook-park/dk-lstm-maritime-pdm:

> Park, D. et al. (2026). A Standards-Aligned Hybrid AI–Digital Twin Framework for Robust Predictive Maintenance under Data Scarcity. *Applied Sciences*, 16, 5303.

The predictor of the previous study (DK-LSTM-v4) is used here as one of three frozen predictors. SC-SP decides for each prediction whether it is used automatically or deferred to human review. The score combines four source-standardized signals:

- input distance (F)
- hidden-representation distance (H)
- conformal interval width (W)
- optimism gap (G)

The threshold is set on a source calibration split. No target labels are used, and the predictor is not retrained.

## Repository Contents

| File | Description |
|---|---|
| `scsp_phase2.py` | Single script that trains the predictors, runs inference and reproduces every table and figure of the manuscript |
| `requirements.txt` | Python dependencies |
| `PAPER_TABLES.md` | All manuscript tables as produced by the script from the final run |
| `figures` | Figs. 1–3 of the manuscript |

## Primary Dataset

The data are not included. Download NASA C-MAPSS (subsets FD001–FD004: `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`) from the NASA Prognostics Center of Excellence data repository (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

- FD001 is the source domain.
- FD002, FD003 and FD004 are the shifted test domains.

## Requirements

Python 3.10 or later, NumPy, pandas, SciPy, scikit-learn, matplotlib, and TensorFlow 2.16 or later. TensorFlow is needed for the training stage only. Install with `pip install -r requirements.txt`.

## Reproducibility

The script has two stages, selected with the environment variable `SCSP_STAGE`.

| Stage | What it does | Hardware |
|---|---|---|
| `all` | Trains 30 predictors (3 architectures × 10 seeds), saves predictions and signals for all scenarios, then analyzes | GPU, about 2 h on an A100 |
| `analyze` | Reads the saved predictions of a run and produces all tables and figures | CPU, a few minutes |

Fixed settings are recorded in `run_manifest.json`. They include:

- source engine split 65/15/20 with split seed 2026
- training seeds 0, 7, 42, 123, 2024, 8649, 76166, 77579, 85257 and 96058
- calibration percentile 95 as the main operating point
- the scenarios and the decision rules

TensorFlow op determinism is enabled, but results on other GPU types or library versions may differ slightly.

### Google Colab

1. Upload `scsp_phase2.py` and the C-MAPSS files to Google Drive, for example:
   ```
   MyDrive/Paper_Experiment/scsp_phase2.py
   MyDrive/Paper_Experiment/CMAPSSData/train_FD001.txt ...
   ```
2. In a new notebook, set the environment variables before running the script:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   import os
   os.environ["SCSP_ROOT_DIR"] = "/content/drive/MyDrive/Paper_Experiment"
   os.environ["SCSP_STAGE"] = "all"        # or "analyze" for an existing run
   # os.environ["SCSP_RUN_TAG"] = "run20260923_v38"   # existing run folder (analyze)
   %run /content/drive/MyDrive/Paper_Experiment/scsp_phase2.py
   ```

### Local machine

```bash
SCSP_STAGE=all SCSP_ROOT_DIR=./work SCSP_DATA_DIR=./CMAPSSData python scsp_phase2.py
```

### Optional settings

| Variable | Meaning |
|---|---|
| `SCSP_SPLIT_SEED` | Seed of the source engine split (default 2026) |
| `SCSP_FAULT_SENSORS` | Masked sensors, e.g. `s_11,s_4,s_12` (default: three sensors most correlated with RUL) |
| `SCSP_OUT_NAME` | Name of the results folder (default `results`) |
| `SCSP_FAST_SMOKE=1` | Tiny settings for a path check; the numbers are not paper results |

## Outputs

The script writes to `<SCSP_ROOT_DIR>/phase2/<RUN_TAG>/results/`.

| Output | Manuscript item |
|---|---|
| `tables/T1_scenarios.csv` | Table 1 |
| `tables/T2_predictors.csv` | Table 2 |
| `tables/Table3_random_deferral.csv` | Table 3 |
| `tables/T4_tests.csv`, `tables/T4_bh1_by_model.csv` | Table 4, Panels A and B; Section 5.2 |
| `tables/Table4C_matched_tests.csv` | Table 4, Panel C |
| `tables/Table5_lower_bound.csv` | Table 5 |
| `tables/Table6_policy.csv` | Table 6 |
| `tables/Table7_ablation_tests.csv`, `tables/Table7_ablation_summary.csv` | Table 7 |
| `tables/Table8_combos_summary.csv`, `tables/Table8_combos_tests.csv` | Table 8 |
| `tables/combined_gate_summary.csv` | Section 5.3 |
| `tables/aurc_auroc_summary.csv` | Section 5.4 |
| `tables/break_even_cost_ratios.csv` | Section 5.5 |
| `tables/T3_gates.csv` | Table A.1 |
| `tables/TableB1_tau.csv` | Table B.1 |
| `figures/Fig1_operating_curves.png`, `Fig2_fault_intensity.png`, `Fig3_expected_cost.png` | Figs. 1–3 |
| `PAPER_TABLES.md` | All tables in the layout of the manuscript |
| `verdicts.json` | Verdicts of the pre-specified hypotheses |
| `run_manifest.json` | Configuration, package versions and output hashes |

Analyses labelled *post-hoc* in the manuscript are in `analysis_posthoc()`. The pre-specified analyses are in `analysis_main()`.

## Notes

- No personal or proprietary information is included.
- All faults are synthetic perturbations of the public C-MAPSS benchmark.
