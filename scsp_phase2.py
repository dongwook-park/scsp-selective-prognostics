# ==============================================================================
# scsp_phase2.py
# Source-Calibrated Selective Prognostics (SC-SP): a deployment-time reliability
# layer for remaining useful life (RUL) prediction under distribution shift.
#
# Single clean script that reproduces every table, figure and number reported in
# the manuscript. It merges four earlier scripts of the same frozen run:
#   sci_journal_phase2_clean_v3.8.py  (training, inference, Tables 1, 2, 4A/B,
#                                      A.1, Figs. 1-3, hypothesis verdicts)
#   scsp_revision_pack_v38.py         (Tables 4C, 5, 8, B.1, combined gate, AURC)
#   scsp_ablation_v38.py              (Table 7)
#   scsp_policy_sim_v38.py            (Table 6)
# The computations are unchanged. Only the figure labels were changed so that
# they use the terms of the manuscript (Aut. HOP@20, medium intensity, dotted
# random-deferral line, scenario names of Table 1).
#
# ------------------------------------------------------------------------------
# Stages (environment variable SCSP_STAGE)
#   all      train 30 predictors, run inference, then analyze   (GPU, ~2 h on A100)
#   compute  train and run inference only
#   analyze  read saved predictions and produce all results    (CPU, minutes)
#
# Typical use (Google Colab)
#   Reproduce the paper from the frozen run (no training):
#       import os
#       os.environ["SCSP_STAGE"] = "analyze"
#       os.environ["SCSP_RUN_TAG"] = "run20260923_v38"
#       %run scsp_phase2.py
#   Full replication with another source split (e.g. split seed 7):
#       os.environ["SCSP_STAGE"] = "all"
#       os.environ["SCSP_SPLIT_SEED"] = "7"
#       os.environ["SCSP_FAULT_SENSORS"] = "s_11,s_4,s_12"   # optional: keep the
#       %run scsp_phase2.py                                    # masked sensors fixed
#
# Environment variables
#   SCSP_STAGE          all | compute | analyze               (default: all)
#   SCSP_ROOT_DIR       root folder                           (default: Colab Drive path)
#   SCSP_DATA_DIR       folder with train_FD001.txt etc.      (searched if unset)
#   SCSP_RUN_TAG        run folder name under ROOT_DIR/phase2 (default: dated tag)
#   SCSP_SPLIT_SEED     seed of the 65/15/20 engine split     (default: 2026, the paper)
#   SCSP_FAULT_SENSORS  comma-separated sensors for M1-M4     (default: top-3 |Spearman|)
#   SCSP_OUT_NAME       results sub-folder                    (default: results)
#   SCSP_FAST_SMOKE     "1" = tiny path check, not paper numbers
#
# Outputs (RUN_DIR/<SCSP_OUT_NAME>)
#   tables/*.csv            one CSV per manuscript table (and per-combination data)
#   figures/Fig1..Fig3.png  manuscript figures
#   PAPER_TABLES.md         all tables formatted as in the manuscript
#   verdicts.json           pre-specified hypothesis verdicts
#   frozen_check.json       comparison with the frozen outputs of run20260923_v38
#   run_manifest.json       configuration, versions and output hashes
#
# Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib;
#               tensorflow (compute stage only).
# Data: NASA C-MAPSS (train/test/RUL_FD001..FD004.txt).
# ==============================================================================

import os
import sys
import json
import time
import random
import hashlib
import warnings
from datetime import datetime, timezone

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, rankdata

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_NAME = "scsp_phase2.py"
ANCHOR_DOC = "v1.1"


# ==============================================================================
# 0. Configuration (pre-specified in the anchor document; do not change)
# ==============================================================================
FAST_SMOKE = os.environ.get("SCSP_FAST_SMOKE", "0") == "1"
STAGE = os.environ.get("SCSP_STAGE", "all")
assert STAGE in ("all", "compute", "analyze"), STAGE
ROOT_DIR = os.environ.get("SCSP_ROOT_DIR", "/content/drive/MyDrive/Paper_Experiment")

# --- data and preprocessing --------------------------------------------------
SEQUENCE_LEN = 30
MAX_RUL = 125.0
EVAL_WIN = 40                       # evaluation windows per test engine (evenly spaced)
SPLIT_SEED = int(os.environ.get("SCSP_SPLIT_SEED", "2026"))   # engine split 65/15/20
FEATURES = ["setting_1", "setting_2",
            "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
            "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
N_FEATURES = len(FEATURES)

# --- predictors (DK-LSTM-v4 settings fixed in the previous study) -------------
MODEL_NAMES = ["DK-LSTM-v4", "GRU", "CNN-LSTM"]
SEEDS = [0, 7, 42, 123, 2024, 8649, 76166, 77579, 85257, 96058]
ENSEMBLE_GROUPS = [[0, 7, 42, 123, 2024], [8649, 76166, 77579, 85257, 96058]]
ENSEMBLE_MIN_MEMBERS = 3
BATCH_SIZE, PREDICT_BATCH = 128, 1024
MAX_EPOCHS, PATIENCE, WARMUP_EPOCHS = 150, 25, 10
LAMBDA_NEG, LAMBDA_UPPER, LAMBDA_SAFETY = 1.0, 2.0, 0.5
SAFETY_OVER, SAFETY_UNDER = 1.5, 0.5
DROPOUT, CLIPNORM = 0.3, 0.5

# --- qualification criterion ---------------------------------------------------
QUAL_RMSE_MAX = 25.0
QUAL_PRED_STD_RATIO = 0.25

# --- uncertainty and signals ---------------------------------------------------
ALPHA_CQR = 0.10
SHRINK_F = 1e-2                     # shrinkage of the input Mahalanobis covariance
SHRINK_H = 5e-2                     # shrinkage of the hidden Mahalanobis covariance

# --- risk and operating points -------------------------------------------------
TAU = 20.0                          # late = (y_pred - y_true) > TAU
TAUS = [10.0, 20.0, 30.0, 40.0]     # Appendix B
OP_PCTS = [80, 90, 95, 99]
OP_MAIN = 95
H_DECISION = 30.0                   # continue-operation horizon (cycles)
COMBINED_BUDGET = 0.05              # source deferral budget of the combined gate

# --- scenarios: perturbations --------------------------------------------------
N_REAL = 5
PERTURB_SEED_BASE = 42
NOISE_SIGMA, DRIFT_MAG = 0.03, 0.05
OUTLIER_RATE, OUTLIER_SPIKE = 0.03, 0.30
MISSING_RATE = 0.30

# --- scenarios: in-range masking faults ----------------------------------------
FAULT_K = 3
FAULT_ONSET = 0.5
FAULT_LEVELS = {                    # bias/drift in units of sigma, gain as factor
    "w": dict(bias=0.5, drift=0.5, gain=0.75),
    "m": dict(bias=1.0, drift=1.0, gain=0.50),
}
_FS = os.environ.get("SCSP_FAULT_SENSORS", "").strip()
FAULT_SENSORS_FIXED = [s.strip() for s in _FS.split(",") if s.strip()] or None

# --- statistics ----------------------------------------------------------------
N_BOOT = 2000
BOOT_SEED = 20260923
BASE_LATE_MIN = 0.01                # B-H1 target: late rate with no gate >= 1%
S0_DEFER_MAX = 0.10                 # B-H1 condition: S0 deferral <= 10%
H2_MIN_SUPPORT = 2                  # B-H2: two of three

# --- gates ---------------------------------------------------------------------
G_NONE, G_RULE, G_P, G_F, G_W, G_SCSP, G_ENS = "G0", "RB", "GP", "GF", "GW", "SC-SP", "GE"
GATES = [G_NONE, G_RULE, G_P, G_F, G_W, G_SCSP, G_ENS]
GATE_COLORS = {G_NONE: "#7f7f7f", G_RULE: "#1f77b4", G_P: "#8c564b", G_F: "#ff7f0e",
               G_W: "#2ca02c", G_SCSP: "#d62728", G_ENS: "#9467bd"}

# --- scenario sets -------------------------------------------------------------
MAIN_SCEN = ["S0", "S1", "S2", "S3", "S4", "S5", "M1m", "M2m", "M3m", "M4"]
H1_CANDIDATES = ["S3", "S4", "S5", "M1w", "M1m", "M2w", "M2m", "M3w", "M3m", "M4"]
H2_SCEN = ["M1m", "M2m", "M3m"]
H2_CONTROL = "M4"
RANDOM_SCEN = ["S0", "S1", "S2", "S3", "M1w", "M1m", "M2w", "M2m", "M3w", "M3m", "M4"]
TAU_SCEN = ["S0", "S3", "M1m", "M2m", "M3m", "M4"]
COST_PANELS = ["S0", "S3", "M2m", "M4"]
SCEN_LABEL = {                      # names of Table 1
    "S0": "In-domain", "S1": "Sensor degradation", "S2": "Compound degradation",
    "S3": "Fault-mode shift (FD003)", "S4": "Operating-condition shift (FD002)",
    "S5": "Combined shift (FD004)",
    "M1w": "Masking bias (weak)", "M1m": "Masking bias (medium)",
    "M2w": "Masking drift (weak)", "M2m": "Masking drift (medium)",
    "M3w": "Gain attenuation (weak)", "M3m": "Gain attenuation (medium)",
    "M4": "Stuck-at (control)",
}
HOP_LABEL = "Aut. HOP@20"           # = hazardous pass-through rate, Eq. (5)

# --- post-hoc signal variants ----------------------------------------------------
SIGNALS = ("F", "H", "W", "G")
ABLATION = {"FULL": ("F", "H", "W", "G"),
            "-F": ("H", "W", "G"), "-H": ("F", "W", "G"),
            "-W": ("F", "H", "G"), "-G": ("F", "H", "W"),
            "F": ("F",), "H": ("H",), "W": ("W",), "G": ("G",)}
ABLATION_REPORT = ["-F", "-H", "-W", "-G", "H", "G"]          # Table 7 columns
COMBOS = {"FULL": ("F", "H", "W", "G"), "F+W+G": ("F", "W", "G"),
          "F+W": ("F", "W"), "F+G": ("F", "G")}               # Table 8

if FAST_SMOKE:
    MAX_EPOCHS, PATIENCE, WARMUP_EPOCHS = 4, 2, 0
    MODEL_NAMES = ["DK-LSTM-v4", "GRU"]
    SEEDS = [0, 7, 42, 123, 2024, 8649]
    ENSEMBLE_GROUPS = [[0, 7, 42], [123, 2024, 8649]]
    EVAL_WIN = 8
    N_REAL = 2
    N_BOOT = 100
    QUAL_RMSE_MAX, QUAL_PRED_STD_RATIO = 1e9, 0.0
    print("[FAST_SMOKE] reduced settings; not paper numbers.")


# ==============================================================================
# 1. Paths
# ==============================================================================
def _mount_drive():
    if os.path.isdir("/content") and not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:                      # pragma: no cover
            print(f"[WARN] Drive mount failed: {e}")


_mount_drive()
_default_tag = datetime.now().strftime("run%Y%m%d_v39")
if SPLIT_SEED != 2026:
    _default_tag += f"_split{SPLIT_SEED}"
RUN_TAG = os.environ.get("SCSP_RUN_TAG") or _default_tag
RUN_DIR = os.path.join(ROOT_DIR, "phase2", RUN_TAG)
MODEL_DIR = os.path.join(RUN_DIR, "models")
PRED_DIR = os.path.join(RUN_DIR, "predictions")
SCEN_NPZ = os.path.join(RUN_DIR, "scenarios.npz")
OUT_DIR = os.path.join(RUN_DIR, os.environ.get("SCSP_OUT_NAME", "results"))
TAB_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")
FROZEN = dict(paper=os.path.join(RUN_DIR, "paper"),
              revision=os.path.join(RUN_DIR, "revision_pack"),
              ablation=os.path.join(RUN_DIR, "ablation"))
for _d in (RUN_DIR, MODEL_DIR, PRED_DIR):
    os.makedirs(_d, exist_ok=True)


def find_data_dir():
    cands = [os.environ.get("SCSP_DATA_DIR", ""), ROOT_DIR,
             os.path.join(ROOT_DIR, "CMAPSSData"), os.path.join(ROOT_DIR, "CMAPSS"),
             "/content/CMAPSSData", "./CMAPSSData", "."]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "train_FD001.txt")):
            return c
    raise FileNotFoundError("train_FD001.txt not found; set SCSP_DATA_DIR.")


def set_all_seeds(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ==============================================================================
# 2. Shared functions (metrics, statistics, signals)
# ==============================================================================
def rmse(y, yp):
    return float(np.sqrt(np.mean((np.asarray(yp) - np.asarray(y)) ** 2))) if len(y) else np.nan


def nasa_mean(y, yp):
    d = np.asarray(yp, float) - np.asarray(y, float)
    return float(np.mean(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def robust_z(values, ref):
    """log1p, then standardize with the median and MAD of the reference (source) values."""
    v = np.log1p(np.maximum(np.asarray(values, np.float64), 0.0))
    r = np.log1p(np.maximum(np.asarray(ref, np.float64), 0.0))
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-8:
        mad = np.std(r) if np.std(r) > 1e-8 else 1.0
    return (v - med) / mad


def variant_score(sig, ref, combo):
    """Equally weighted mean of source robust z-scores. SC-SP = combo (F, H, W, G)."""
    return (1.0 / len(combo)) * sum(robust_z(sig[k], ref[k]) for k in combo)


def scsp_score(sig, ref):
    return variant_score(sig, ref, SIGNALS)


def quantile_gate(cal_score, score, pct):
    """Accept (True) when the score does not exceed the pct-th percentile of the calibration scores."""
    return score <= np.quantile(cal_score, pct / 100.0)


def rule_bank_thresholds(cal_R, pct):
    """Thresholds of the OR of four rules. A common quantile level is found by bisection so that
    the calibration deferral rate is at most 1 - pct/100."""
    target = 1.0 - pct / 100.0
    lo_q, hi_q = 0.5, 1.0
    for _ in range(40):
        mid = 0.5 * (lo_q + hi_q)
        rate = (cal_R > np.quantile(cal_R, mid, axis=0)).any(axis=1).mean()
        lo_q, hi_q = (mid, hi_q) if rate > target else (lo_q, mid)
    return np.quantile(cal_R, hi_q, axis=0)


def gate_metrics(acc, y, yp):
    late = (yp - y) > TAU
    cont = acc & (yp > H_DECISION)                  # automatic continue-operation
    fc = cont & (y <= H_DECISION)                   # ... although failure within H
    n_late, n_cont = int(late.sum()), int(cont.sum())
    return dict(defer=float(1 - acc.mean()), ahop=float((acc & late).mean()),
                late0=float(late.mean()),
                capture=(float((~acc & late).sum() / n_late) if n_late else np.nan),
                acc_rmse=rmse(y[acc], yp[acc]),
                fcr=float(fc.mean()),
                fcr_acc=(float(fc.sum() / n_cont) if n_cont else np.nan))


def met(acc, late):
    nl = late.sum()
    return dict(defer=float(1 - acc.mean()), ahop=float((acc & late).mean()), late0=float(late.mean()),
                capture=float((~acc & late).sum() / nl) if nl else np.nan)


def topk_defer(score, k):
    """Defer the k windows with the highest score (stable sort for ties)."""
    acc = np.ones(len(score), bool)
    if k > 0:
        acc[np.argsort(-score, kind="stable")[:k]] = False
    return acc


def aurc_auroc(score, late):
    o = np.argsort(score, kind="stable")
    l_ = late[o].astype(float)
    sel = np.cumsum(l_) / np.arange(1, len(l_) + 1)
    n1, n0 = late.sum(), (~late).sum()
    auc = (rankdata(score)[late].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    return float(sel.mean()), float(auc)


def bh(pvals):
    p = np.asarray(pvals, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]; m = len(pv); o = np.argsort(pv)
    adj = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def paired_wilcoxon(a, b):
    """(median of a-b, p, n); nan p when fewer than three pairs or all differences are zero."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.allclose(a, b):
        return (float(np.median(a - b)) if len(a) else np.nan), np.nan, len(a)
    return float(np.median(a - b)), float(wilcoxon(a, b).pvalue), len(a)


def one_sample_wilcoxon(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or np.allclose(x, 0):
        return (float(np.median(x)) if len(x) else np.nan), np.nan
    return float(np.median(x)), float(wilcoxon(x).pvalue)


def scen_of(key):
    return key.split("_r")[0]


def tag_of(name, seed):
    return f"{name.replace('-', '_')}_seed{seed}"


# ==============================================================================
# 3. Compute stage: data, scenarios, predictors, saved predictions
# ==============================================================================
COLS = (["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"]
        + [f"s_{i}" for i in range(1, 22)])


def stage_compute():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.preprocessing import MinMaxScaler
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    print(f"TF {tf.__version__}  GPU={tf.config.list_physical_devices('GPU')}")
    if not FAST_SMOKE and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("No GPU. To analyze saved predictions only, set SCSP_STAGE=analyze.")

    data_dir = find_data_dir()

    def load_subset(fd):
        rd = lambda n: pd.read_csv(os.path.join(data_dir, n), sep=r"\s+", header=None)
        tr = rd(f"train_{fd}.txt"); tr.columns = COLS
        te = rd(f"test_{fd}.txt"); te.columns = COLS
        rul = rd(f"RUL_{fd}.txt").iloc[:, 0].values.astype(float)
        return tr, te, rul

    def add_rul(df):
        life = df.groupby("unit_nr")["time_cycles"].transform("max")
        df = df.copy()
        df["RUL"] = (life - df["time_cycles"]).clip(upper=MAX_RUL)
        return df

    def make_sequences(df):
        """Training/calibration windows (stride 1); target = RUL at the step after the window."""
        X, y = [], []
        for _, unit in df.groupby("unit_nr", sort=True):
            v, r = unit[FEATURES].values, unit["RUL"].values
            for i in range(len(unit) - SEQUENCE_LEN):
                X.append(v[i:i + SEQUENCE_LEN]); y.append(r[i + SEQUENCE_LEN])
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    def window_ends(L):
        if L < SEQUENCE_LEN:
            return np.asarray([], int)
        return np.unique(np.round(np.linspace(SEQUENCE_LEN, L, EVAL_WIN)).astype(int))

    train_fd1, test_fd1, rul_fd1 = load_subset("FD001")
    TEST = {"FD001": (test_fd1, rul_fd1)}
    for fd in ("FD002", "FD003", "FD004"):
        _, te, r = load_subset(fd)
        TEST[fd] = (te, r)
    train_fd1 = add_rul(train_fd1)
    scaler = MinMaxScaler()
    train_fd1[FEATURES] = scaler.fit_transform(train_fd1[FEATURES])

    units = np.random.default_rng(SPLIT_SEED).permutation(train_fd1["unit_nr"].unique())
    n = len(units)
    a, b = int(round(n * 0.65)), int(round(n * 0.65)) + int(round(n * 0.15))
    fit_df = train_fd1[train_fd1["unit_nr"].isin(units[:a])]
    tune_df = train_fd1[train_fd1["unit_nr"].isin(units[a:b])]
    cal_df = train_fd1[train_fd1["unit_nr"].isin(units[b:])]
    x_fit, y_fit = make_sequences(fit_df)
    x_tune, y_tune = make_sequences(tune_df)
    x_cal, y_cal = make_sequences(cal_df)
    print(f"split seed {SPLIT_SEED}: engines fit/tune/cal = {a}/{b - a}/{n - b}  "
          f"windows = {len(y_fit)}/{len(y_tune)}/{len(y_cal)}")

    def build_pool(test_df, rul_raw, transform=None):
        """Evenly spaced windows per test engine; transform(vals[L, F]) acts on the scaled series
        before windowing. Target: min(r_last + (L - e), MAX_RUL)."""
        df = test_df.copy()
        df[FEATURES] = scaler.transform(df[FEATURES])
        X, y, U = [], [], []
        for idx, uid in enumerate(sorted(df["unit_nr"].unique())):
            unit = df[df["unit_nr"] == uid]
            L = len(unit)
            ends = window_ends(L)
            if len(ends) == 0:
                continue
            vals = unit[FEATURES].values.astype(np.float32).copy()
            if transform is not None:
                vals = transform(vals)
            for e in ends:
                X.append(vals[e - SEQUENCE_LEN:e])
                y.append(min(float(rul_raw[idx]) + (L - e), MAX_RUL))
                U.append(uid)
        return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(U)

    # ---- scenarios ------------------------------------------------------------
    FIT_MEDIAN = np.median(fit_df[FEATURES].values, axis=0).astype(np.float32)
    FIT_MEAN = fit_df[FEATURES].values.mean(axis=0).astype(np.float32)
    FIT_STD = fit_df[FEATURES].values.std(axis=0).astype(np.float32)
    rho = {j: spearmanr(fit_df[f].values, fit_df["RUL"].values).correlation
           for j, f in enumerate(FEATURES) if f.startswith("s_")}
    if FAULT_SENSORS_FIXED:
        fault_idx = [FEATURES.index(s) for s in FAULT_SENSORS_FIXED]
    else:
        fault_idx = sorted(rho, key=lambda j: -abs(rho[j]))[:FAULT_K]
    fault_dir = np.asarray([np.sign(rho[j]) for j in fault_idx], np.float32)   # healthy direction
    fault_sensors = [FEATURES[j] for j in fault_idx]
    print(f"fault sensors = {fault_sensors}  healthy direction = {fault_dir.tolist()}")

    def fault_transform(kind, level):
        p = FAULT_LEVELS.get(level, {})

        def tf_(vals):
            L = len(vals); t0 = int(np.floor(FAULT_ONSET * L))
            out = vals.copy()
            for c, j in enumerate(fault_idx):
                seg = out[t0:, j]
                if kind == "M1":
                    seg += fault_dir[c] * p["bias"] * FIT_STD[j]
                elif kind == "M2":
                    ramp = np.arange(1, L - t0 + 1, dtype=np.float32) / max(L - t0, 1)
                    seg += fault_dir[c] * p["drift"] * FIT_STD[j] * ramp
                elif kind == "M3":
                    seg[:] = FIT_MEAN[j] + p["gain"] * (seg - FIT_MEAN[j])
                elif kind == "M4":
                    seg[:] = vals[t0, j]
                out[t0:, j] = seg
            return out
        return tf_

    def missing_transform(seed):
        """Causal missingness followed by forward filling."""
        rng = np.random.default_rng(seed)

        def tf_(vals):
            drop = rng.random(vals.shape) < MISSING_RATE
            last = FIT_MEDIAN.copy()
            out = vals.copy()
            for t in range(len(out)):
                last = np.where(~drop[t], out[t], last)
                out[t] = last
            return out
        return tf_

    def add_noise(X, seed):
        return (X + np.random.default_rng(seed).normal(0, NOISE_SIGMA, X.shape)).astype(np.float32)

    def add_drift(X):
        return (X + np.linspace(0, DRIFT_MAG, X.shape[1], dtype=np.float32)[None, :, None]).astype(np.float32)

    def add_outliers(X, seed):
        rng = np.random.default_rng(seed)
        mask = rng.random(X.shape) < OUTLIER_RATE
        sign = rng.choice(np.asarray([-1.0, 1.0], np.float32), size=X.shape)
        return (X + mask * sign * OUTLIER_SPIKE).astype(np.float32)

    def build_scenarios():
        sc = {}
        base = build_pool(*TEST["FD001"])
        sc["S0"] = base
        for r in range(N_REAL):
            rs = PERTURB_SEED_BASE + 3 * r
            sc[f"S1_r{r}"] = (add_drift(add_noise(base[0], rs)), base[1], base[2])
            Xm, ym, Um = build_pool(*TEST["FD001"], transform=missing_transform(rs))
            assert np.array_equal(Um, base[2]) and np.array_equal(ym, base[1])
            sc[f"S2_r{r}"] = (add_outliers(add_drift(add_noise(Xm, rs + 1)), rs + 2), base[1], base[2])
        sc["S3"] = build_pool(*TEST["FD003"])
        sc["S4"] = build_pool(*TEST["FD002"])
        sc["S5"] = build_pool(*TEST["FD004"])
        for kind in ("M1", "M2", "M3"):
            for lv in ("w", "m"):
                sc[f"{kind}{lv}"] = build_pool(*TEST["FD001"], transform=fault_transform(kind, lv))
        sc["M4"] = build_pool(*TEST["FD001"], transform=fault_transform("M4", None))
        return sc

    # ---- model-independent signals: input distance F and rule scores R ----------
    def mahalanobis2(Z, mu, inv):
        d = np.asarray(Z, np.float64) - mu
        return np.einsum("ni,ij,nj->n", d, inv, d)

    def shrunk_inv_cov(Z, shrink):
        cov = np.atleast_2d(np.cov(np.asarray(Z, np.float64).T, ddof=1))
        lam = shrink * np.trace(cov) / cov.shape[0]
        return np.linalg.pinv(cov + lam * np.eye(cov.shape[0]))

    Fcal = x_cal.mean(axis=1)
    F_MU, F_INV = Fcal.mean(axis=0), shrunk_inv_cov(Fcal, SHRINK_F)
    signal_F = lambda X: mahalanobis2(X.mean(axis=1), F_MU, F_INV)

    EPS = 1e-4                       # rule references from the fit split, thresholds from calibration
    R_MIN, R_MAX = x_fit.min(axis=(0, 1)), x_fit.max(axis=(0, 1))
    R_DIFF_SD = np.abs(np.diff(x_fit, axis=1)).reshape(-1, N_FEATURES).std(axis=0) + EPS
    R_WSD_MED = np.median(x_fit.std(axis=1), axis=0)
    wm = x_fit.mean(axis=1)
    R_MEAN_MED = np.median(wm, axis=0)
    R_MEAN_MAD = np.median(np.abs(wm - R_MEAN_MED), axis=0) * 1.4826 + EPS

    def rule_scores(X):
        """R1 out-of-range share, R2 max rate of change, R3 stuck-at degree, R4 max univariate |z|."""
        r1 = ((X < R_MIN) | (X > R_MAX)).mean(axis=(1, 2))
        r2 = (np.abs(np.diff(X, axis=1)) / R_DIFF_SD).max(axis=(1, 2))
        r3 = np.log((R_WSD_MED + EPS) / (X.std(axis=1) + EPS)).max(axis=1)
        r4 = (np.abs(X.mean(axis=1) - R_MEAN_MED) / R_MEAN_MAD).max(axis=1)
        return np.stack([r1, r2, r3, r4], axis=1).astype(np.float32)

    # ---- predictors -------------------------------------------------------------
    def dk_loss_fn_v4(y_true, y_pred):
        err = y_pred - y_true
        mse = tf.reduce_mean(tf.square(err))
        safety = tf.reduce_mean(tf.where(err > 0, tf.square(err) * SAFETY_OVER,
                                         tf.square(err) * SAFETY_UNDER))
        neg = tf.reduce_mean(tf.square(tf.nn.relu(-y_pred)))
        upper = tf.reduce_mean(tf.square(tf.nn.relu(y_pred - MAX_RUL)))
        return mse + LAMBDA_SAFETY * safety + LAMBDA_NEG * neg + LAMBDA_UPPER * upper

    def build_model(name):
        inp = layers.Input(shape=(SEQUENCE_LEN, N_FEATURES))
        if name == "DK-LSTM-v4":
            x = layers.Dropout(DROPOUT)(layers.LSTM(64, return_sequences=True)(inp))
            h = layers.LSTM(32, name="hidden_state")(x)
            loss = dk_loss_fn_v4
        elif name == "GRU":
            x = layers.Dropout(DROPOUT)(layers.GRU(64, return_sequences=True)(inp))
            h = layers.GRU(32, name="hidden_state")(x)
            loss = "mse"
        elif name == "CNN-LSTM":
            x = layers.Dropout(DROPOUT)(layers.Conv1D(32, 3, padding="same", activation="relu")(inp))
            x = layers.Dropout(DROPOUT)(layers.LSTM(64, return_sequences=True)(x))
            h = layers.LSTM(32, name="hidden_state")(x)
            loss = "mse"
        else:
            raise ValueError(name)
        x = layers.Dense(32, activation="relu")(layers.Dropout(DROPOUT)(h))
        out = layers.Dense(1, activation="linear")(x)
        m = models.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=CLIPNORM), loss=loss, metrics=["mae"])
        return m

    def cosine_lr(epoch, lr):
        if epoch < WARMUP_EPOCHS:
            return 1e-3
        p = min((epoch - WARMUP_EPOCHS) / 60.0, 1.0)
        return float(1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + np.cos(np.pi * p)))

    def train_point_model(name, seed):
        path = os.path.join(MODEL_DIR, f"point_{tag_of(name, seed)}.keras")
        if os.path.exists(path + ".done"):
            return tf.keras.models.load_model(path, compile=False)
        set_all_seeds(seed)
        m = build_model(name)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE,
                                                restore_best_weights=True),
               tf.keras.callbacks.LearningRateScheduler(cosine_lr)]
        m.fit(x_fit, y_fit, validation_data=(x_tune, y_tune), epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
        m.save(path)
        open(path + ".done", "w").write(datetime.now().isoformat())
        return m

    def predict_rul(m, X):
        return np.clip(m.predict(X, batch_size=PREDICT_BATCH, verbose=0).reshape(-1), 0, MAX_RUL)

    def pinball(q):
        def loss(y, p):
            e = y - p
            return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))
        return loss

    def fit_cqr(hidden, seed):
        """Quantile heads trained on the fit split, split-conformal correction on the calibration split."""
        H_fit = hidden.predict(x_fit, batch_size=PREDICT_BATCH, verbose=0)
        H_tune = hidden.predict(x_tune, batch_size=PREDICT_BATCH, verbose=0)
        H_cal = hidden.predict(x_cal, batch_size=PREDICT_BATCH, verbose=0)
        heads = []
        for k, q in enumerate((ALPHA_CQR / 2, 1 - ALPHA_CQR / 2)):
            set_all_seeds(seed + 100000 + k)
            inp = layers.Input(shape=(H_fit.shape[1],))
            o = layers.Dense(1)(layers.Dense(16, activation="relu")(layers.Dense(32, activation="relu")(inp)))
            hm = models.Model(inp, o)
            hm.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=pinball(q))
            hm.fit(H_fit, y_fit, validation_data=(H_tune, y_tune), epochs=MAX_EPOCHS,
                   batch_size=BATCH_SIZE, verbose=0,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                               restore_best_weights=True)])
            heads.append(hm)

        def raw(H):
            a_ = heads[0].predict(H, batch_size=PREDICT_BATCH, verbose=0).reshape(-1)
            b_ = heads[1].predict(H, batch_size=PREDICT_BATCH, verbose=0).reshape(-1)
            return np.minimum(a_, b_), np.maximum(a_, b_)

        lo_c, hi_c = raw(H_cal)
        E = np.maximum(lo_c - y_cal, y_cal - hi_c)
        k = min(max(int(np.ceil((len(E) + 1) * (1 - ALPHA_CQR))) - 1, 0), len(E) - 1)
        Q = float(np.sort(E)[k])

        def interval(H):
            lo, hi = raw(H)
            lo = np.clip(lo - Q, 0, MAX_RUL); hi = np.clip(hi + Q, 0, MAX_RUL)
            return lo, np.maximum(hi, lo)
        return interval, Q

    def hidden_distance_fn(hidden):
        """Hidden-representation Mahalanobis distance H; reference = calibration split."""
        Hc = hidden.predict(x_cal, batch_size=PREDICT_BATCH, verbose=0)
        mu, sd = Hc.mean(axis=0), Hc.std(axis=0)
        keep = sd > max(1e-6, np.percentile(sd, 10) * 0.1)
        std = lambda H: (H[:, keep] - mu[keep]) / (sd[keep] + 1e-8)
        Zc = std(Hc)
        muZ, inv = Zc.mean(axis=0), shrunk_inv_cov(Zc, SHRINK_H)
        return lambda H: mahalanobis2(std(H), muZ, inv)

    # ---- run --------------------------------------------------------------------
    sc = build_scenarios()
    store, meta = {}, []
    store["CAL__y"] = y_cal
    store["CAL__F"] = signal_F(x_cal).astype(np.float32)
    store["CAL__R"] = rule_scores(x_cal)
    for key, (X, y, U) in sc.items():
        store[f"{key}__y"], store[f"{key}__unit"] = y, U
        store[f"{key}__F"] = signal_F(X).astype(np.float32)
        store[f"{key}__R"] = rule_scores(X)
        meta.append(dict(key=key, scen=scen_of(key), n=len(y), n_units=len(np.unique(U)),
                         oob_frac=float(((X < 0) | (X > 1)).mean())))
    np.savez_compressed(SCEN_NPZ, **store)
    pd.DataFrame(meta).to_csv(os.path.join(RUN_DIR, "scenario_meta.csv"), index=False)
    json.dump(dict(split_seed=SPLIT_SEED, fault_sensors=fault_sensors,
                   fault_healthy_direction=fault_dir.tolist(),
                   engines_fit_tune_cal=[a, b - a, n - b]),
              open(os.path.join(RUN_DIR, "compute_info.json"), "w"), indent=2)

    qual_rows = []
    for name in MODEL_NAMES:
        for seed in SEEDS:
            out_path = os.path.join(PRED_DIR, f"{tag_of(name, seed)}.npz")
            qpath = out_path.replace(".npz", "_qual.json")
            if os.path.exists(out_path) and os.path.exists(qpath):
                qual_rows.append(json.load(open(qpath)))
                print(f"[CACHE] {name} seed={seed}")
                continue
            t0 = time.time()
            m = train_point_model(name, seed)
            hidden = models.Model(m.input, m.get_layer("hidden_state").output)
            interval, Q = fit_cqr(hidden, seed)
            hdist = hidden_distance_fn(hidden)
            yp_c = predict_rul(m, x_cal)
            q = dict(model=name, seed=seed, calib_rmse=rmse(y_cal, yp_c),
                     pred_std=float(np.std(yp_c)), true_std=float(np.std(y_cal)), Q_cqr=Q)
            q["qualified"] = bool(q["calib_rmse"] <= QUAL_RMSE_MAX and
                                  q["pred_std"] >= QUAL_PRED_STD_RATIO * q["true_std"])
            res = {}
            for key, X in [("CAL", x_cal)] + [(k, v[0]) for k, v in sc.items()]:
                H = hidden.predict(X, batch_size=PREDICT_BATCH, verbose=0)
                lo, hi = interval(H)
                res[f"{key}__yp"] = predict_rul(m, X).astype(np.float32)
                res[f"{key}__lo"], res[f"{key}__hi"] = lo.astype(np.float32), hi.astype(np.float32)
                res[f"{key}__H"] = hdist(H).astype(np.float32)
            np.savez_compressed(out_path, **res)
            json.dump(q, open(qpath, "w"))
            qual_rows.append(q)
            print(f"[DONE] {name} seed={seed} calib_rmse={q['calib_rmse']:.2f} "
                  f"qualified={q['qualified']} {(time.time() - t0) / 60:.1f}min")
            tf.keras.backend.clear_session()
    pd.DataFrame(qual_rows).to_csv(os.path.join(RUN_DIR, "qualification.csv"), index=False)
    return dict(tensorflow=tf.__version__)


# ==============================================================================
# 4. Analyze stage: loading and window-level quantities
# ==============================================================================
class Run:
    """Saved scenarios and predictions of one run."""

    def __init__(self):
        self.S = np.load(SCEN_NPZ)
        self.keys = sorted({k.split("__")[0] for k in self.S.files} - {"CAL"})
        qual = pd.read_csv(os.path.join(RUN_DIR, "qualification.csv"))
        self.combos = [(r.model, int(r.seed)) for r in qual.itertuples()]
        self.qual = [(r.model, int(r.seed)) for r in qual.itertuples() if bool(r.qualified)]
        self.QUAL = set(self.qual)
        self.P = {c: dict(np.load(os.path.join(PRED_DIR, f"{tag_of(*c)}.npz"))) for c in self.combos}
        self.RB_THR = {pct: rule_bank_thresholds(self.S["CAL__R"], pct) for pct in OP_PCTS}
        # ensemble disagreement: std of the point predictions of the qualified members of a seed group
        self.ens = {}
        for name in sorted({c[0] for c in self.combos}):
            for grp in ENSEMBLE_GROUPS:
                mem = [(name, s) for s in grp if (name, s) in self.QUAL]
                if len(mem) < ENSEMBLE_MIN_MEMBERS:
                    continue
                for key in ["CAL"] + self.keys:
                    sd = np.std(np.stack([self.P[c][f"{key}__yp"] for c in mem]), axis=0)
                    for s in grp:
                        self.ens[((name, s), key)] = sd

    def signals(self, c, key):
        p = self.P[c]
        yp, lo, hi = p[f"{key}__yp"], p[f"{key}__lo"], p[f"{key}__hi"]
        return dict(F=self.S[f"{key}__F"], H=p[f"{key}__H"], W=hi - lo,
                    G=np.maximum(0, yp - lo), P=yp)

    def rule_accept(self, key, thr):
        return ~(self.S[f"{key}__R"] > thr).any(axis=1)


def add_derived(df):
    df = df.copy()
    df["random_ahop"] = (1.0 - df.defer) * df.late0          # random deferral at the same rate
    df["cond_hop"] = np.where(df.defer < 1.0, df.ahop / (1.0 - df.defer).clip(lower=1e-12), np.nan)
    df["lift"] = np.where(df.defer > 0.0, df.capture / df.defer.clip(lower=1e-12), np.nan)
    return df


# ==============================================================================
# 5. Pre-specified analysis: Tables 1, 2, 4 (A, B), A.1, Figs. 1-3, verdicts
# ==============================================================================
def analysis_main(run, save):
    S, keys, combos, QUAL = run.S, run.keys, run.combos, run.QUAL
    rows, ctx_rows = [], []
    win_sum, win_cnt = {}, {}                       # for the engine-level bootstrap
    for c in combos:
        ref = run.signals(c, "CAL")
        cal_scsp = scsp_score(ref, ref)
        for key in keys:
            y, yp = S[f"{key}__y"], run.P[c][f"{key}__yp"]
            lo, hi = run.P[c][f"{key}__lo"], run.P[c][f"{key}__hi"]
            sig = run.signals(c, key)
            ctx_rows.append(dict(model=c[0], seed=c[1], key=key, scen=scen_of(key),
                                 rmse=rmse(y, yp), nasa=nasa_mean(y, yp),
                                 late=float(((yp - y) > TAU).mean()),
                                 coverage=float(((y >= lo) & (y <= hi)).mean()),
                                 mpiw=float((hi - lo).mean())))
            scores = {G_P: (ref["P"], sig["P"]), G_F: (ref["F"], sig["F"]),
                      G_W: (ref["W"], sig["W"]), G_SCSP: (cal_scsp, scsp_score(sig, ref))}
            if (c, key) in run.ens:
                scores[G_ENS] = (run.ens[(c, "CAL")], run.ens[(c, key)])
            late = (yp - y) > TAU
            for pct in OP_PCTS:
                acc = {G_NONE: np.ones(len(y), bool), G_RULE: run.rule_accept(key, run.RB_THR[pct])}
                for g, (cs, sc) in scores.items():
                    acc[g] = quantile_gate(cs, sc, pct)
                for g, a in acc.items():
                    rows.append(dict(model=c[0], seed=c[1], key=key, scen=scen_of(key),
                                     gate=g, op=pct, **gate_metrics(a, y, yp)))
                    if pct == OP_MAIN and c in QUAL:
                        k2 = (g, key)
                        win_sum[k2] = win_sum.get(k2, 0.0) + (a & late).astype(float)
                        win_cnt[k2] = win_cnt.get(k2, 0) + 1
    RAW = pd.DataFrame(rows)
    CTX_RAW = pd.DataFrame(ctx_rows)
    save(RAW, "gate_metrics_all.csv")
    save(CTX_RAW, "predictor_metrics_all.csv")

    R = RAW.groupby(["model", "seed", "scen", "gate", "op"], as_index=False).mean(numeric_only=True)
    CTX = CTX_RAW.groupby(["model", "seed", "scen"], as_index=False).mean(numeric_only=True)
    R["qualified"] = [(m, s) in QUAL for m, s in zip(R.model, R.seed)]
    CTX["qualified"] = [(m, s) in QUAL for m, s in zip(CTX.model, CTX.seed)]
    rq = R[R.qualified]

    # ---- Table 1 --------------------------------------------------------------
    meta = pd.read_csv(os.path.join(RUN_DIR, "scenario_meta.csv"))
    base = rq[(rq.gate == G_NONE) & (rq.op == OP_MAIN)].groupby("scen")["late0"].mean()
    T1 = meta.groupby("scen", as_index=False)[["n", "n_units", "oob_frac"]].mean()
    T1["label"] = T1.scen.map(SCEN_LABEL)
    T1["base_late"] = T1.scen.map(base)
    T1 = T1[T1.scen.isin(MAIN_SCEN)].set_index("scen").loc[MAIN_SCEN].reset_index()
    T1 = save(T1[["scen", "label", "n", "n_units", "oob_frac", "base_late"]], "T1_scenarios.csv")

    # ---- Table 2 --------------------------------------------------------------
    cq = CTX[CTX.qualified & CTX.scen.isin(MAIN_SCEN)]
    T2 = cq.groupby(["model", "scen"]).agg(
        n_seeds=("seed", "nunique"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        nasa=("nasa", "mean"), late=("late", "mean"),
        coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
        mpiw=("mpiw", "mean")).reset_index()
    T2["scen"] = pd.Categorical(T2.scen, MAIN_SCEN, ordered=True)
    T2 = save(T2.sort_values(["model", "scen"]), "T2_predictors.csv")
    shift = T2[T2.scen != "S0"]
    ah1 = {m: dict(min_coverage=float(g.coverage.min()), scen=str(g.loc[g.coverage.idxmin(), "scen"]))
           for m, g in shift.groupby("model")}

    # ---- Table A.1 (named T3 in the run) ---------------------------------------
    MET = ["defer", "ahop", "late0", "capture", "acc_rmse", "fcr", "fcr_acc"]
    agg = add_derived(rq.groupby(["scen", "gate", "op"], as_index=False)[MET].mean())
    ahop0 = agg[agg.gate == G_NONE].set_index(["scen", "op"])["ahop"]
    agg["reduction"] = 1.0 - agg.ahop / [ahop0.get((s, o), np.nan) for s, o in zip(agg.scen, agg.op)]
    T3 = agg[(agg.op == OP_MAIN) & agg.scen.isin(MAIN_SCEN)].copy()
    T3["scen"] = pd.Categorical(T3.scen, MAIN_SCEN, ordered=True)
    T3["gate"] = pd.Categorical(T3.gate, GATES, ordered=True)
    T3 = save(T3.sort_values(["scen", "gate"])[
        ["scen", "gate", "defer", "ahop", "random_ahop", "reduction", "cond_hop",
         "capture", "lift", "acc_rmse", "fcr", "fcr_acc"]], "T3_gates.csv")

    # ---- Table 4, Panels A and B -------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)

    def unit_boot(scen, g1, g2):
        ks = [k for k in keys if scen_of(k) == scen]
        if any((g, k) not in win_sum for g in (g1, g2) for k in ks):
            return np.nan, np.nan, np.nan
        d = np.mean([win_sum[(g1, k)] / win_cnt[(g1, k)] - win_sum[(g2, k)] / win_cnt[(g2, k)]
                     for k in ks], axis=0)
        _, inv = np.unique(S[f"{ks[0]}__unit"], return_inverse=True)
        du = np.bincount(inv, weights=d) / np.bincount(inv)
        bs = du[rng.integers(0, len(du), (N_BOOT, len(du)))].mean(axis=1)
        return float(du.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def test_family(df, pairs, family, boot):
        if not pairs:                               # possible only in FAST_SMOKE
            return pd.DataFrame(columns=["family", "scen", "gate", "vs", "n", "median_diff", "p", "q",
                                         "supports", "unit_mean_diff", "unit_lo95", "unit_hi95"])
        out = []
        for scen, g1, g2 in pairs:
            a = df[(df.scen == scen) & (df.gate == g1)].set_index(["model", "seed"])["ahop"]
            b = df[(df.scen == scen) & (df.gate == g2)].set_index(["model", "seed"])["ahop"]
            j = a.index.intersection(b.index)
            d, p, n = paired_wilcoxon(a[j].values, b[j].values)
            row = dict(family=family, scen=scen, gate=g1, vs=g2, n=n, median_diff=d, p=p)
            if boot:
                row["unit_mean_diff"], row["unit_lo95"], row["unit_hi95"] = unit_boot(scen, g1, g2)
            out.append(row)
        out = pd.DataFrame(out)
        out["q"] = bh(out.p.values)
        out["supports"] = (out.q < 0.05) & (out.median_diff < 0)
        first = ["family", "scen", "gate", "vs", "n", "median_diff", "p", "q", "supports"]
        return out[first + [c_ for c_ in out.columns if c_ not in first]]

    def run_tests(df, boot):
        df = df[df.op == OP_MAIN]
        s0_defer = float(df[(df.scen == "S0") & (df.gate == G_SCSP)].defer.mean())
        b0 = df[df.gate == G_NONE].groupby("scen").late0.mean()
        h1 = [s for s in H1_CANDIDATES if s in b0.index and b0[s] >= BASE_LATE_MIN]
        fams = [test_family(df, [(s, G_SCSP, G_NONE) for s in h1], "B-H1", boot),
                test_family(df, [(s, G_SCSP, G_RULE) for s in H2_SCEN], "B-H2", boot),
                test_family(df, [(H2_CONTROL, G_SCSP, G_RULE)], "B-H2 control", boot)]
        for g in (G_P, G_F, G_W, G_ENS):
            fams.append(test_family(df, [(s, G_SCSP, g) for s in MAIN_SCEN], f"vs {g}", boot))
        t1, t2 = fams[0], fams[1]
        verdict = dict(
            s0_defer_scsp=s0_defer,
            b_h1=dict(target=h1, n_support=int(t1.supports.sum()),
                      supported=bool(len(h1) and t1.supports.sum() > len(h1) / 2
                                     and s0_defer <= S0_DEFER_MAX)),
            b_h2=dict(n_support=int(t2.supports.sum()),
                      supported=bool(t2.supports.sum() >= H2_MIN_SUPPORT)))
        return pd.concat(fams, ignore_index=True), verdict

    T4, verdict_q = run_tests(rq, boot=True)
    T4 = save(T4, "T4_tests.csv")
    _, verdict_all = run_tests(R, boot=False)       # sensitivity to the qualification criterion

    # per model family, SC-SP - G0 (Section 5.2, one sentence)
    r95 = rq[rq.op == OP_MAIN]
    fam_rows = []
    for (m, s), g in r95[r95.scen.isin(H1_CANDIDATES)].groupby(["model", "scen"]):
        a = g[g.gate == G_SCSP].set_index("seed").ahop
        b = g[g.gate == G_NONE].set_index("seed").ahop
        j = a.index.intersection(b.index)
        fam_rows.append(dict(model=m, scen=s, n=len(j), median_diff=float(np.median(a[j] - b[j])),
                             n_negative=int(((a[j] - b[j]) < 0).sum())))
    save(pd.DataFrame(fam_rows), "T4_bh1_by_model.csv")

    # ---- Figures ------------------------------------------------------------------
    figures = make_figures(agg, T3)

    verdict = dict(anchor=ANCHOR_DOC, n_qualified=len(QUAL), n_combos=len(combos),
                   A_H1=dict(supported=all(v["min_coverage"] < 0.80 for v in ah1.values()), per_model=ah1),
                   qualified=verdict_q, all_seeds=verdict_all)
    return dict(RAW=RAW, R=R, rq=rq, agg=agg, T1=T1, T2=T2, T3=T3, T4=T4,
                verdict=verdict, figures=figures)


def make_figures(agg, T3):
    out = []
    pct = lambda v: np.asarray(v, float) * 100.0

    # ---- Fig. 1: operating curves --------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6))
    for ax, scen in zip(axes.ravel(), MAIN_SCEN):
        d = agg[agg.scen == scen]
        l0 = float(d[(d.gate == G_NONE) & (d.op == OP_MAIN)].ahop.iloc[0])
        ax.plot([0, 100], [l0 * 100, 0], ls=":", color="0.55", lw=1.2, label="random deferral")
        ax.scatter([0], [l0 * 100], color=GATE_COLORS[G_NONE], s=25, zorder=3, label=G_NONE)
        for g in GATES[1:]:
            e = d[d.gate == g].sort_values("defer")
            if not len(e):
                continue
            ax.plot(pct(e.defer), pct(e.ahop), marker="o", ms=3, color=GATE_COLORS[g],
                    lw=2.2 if g == G_SCSP else 1.0, label=g)
            m = e[e.op == OP_MAIN]
            ax.scatter(pct(m.defer), pct(m.ahop), s=45, color=GATE_COLORS[g], edgecolor="k", zorder=4)
        ax.set_xlim(-2, 102); ax.set_ylim(bottom=0)
        ax.set_title(f"{scen}: {SCEN_LABEL[scen]}", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Deferral rate (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{HOP_LABEL} (%)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "Fig1_operating_curves.png"); fig.savefig(p, dpi=300); out.append(p)
    plt.close(fig)

    # ---- Fig. 2: masking-fault intensity ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, kind in zip(axes, ["M1", "M2", "M3"]):
        for g in GATES[1:]:
            ys = []
            for s in ["S0", f"{kind}w", f"{kind}m"]:
                v = agg[(agg.scen == s) & (agg.gate == g) & (agg.op == OP_MAIN)].reduction
                ys.append(float(v.iloc[0]) * 100 if len(v) else np.nan)
            ax.plot([0, 1, 2], ys, marker="o", color=GATE_COLORS[g], label=g,
                    lw=2.2 if g == G_SCSP else 1.0)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["none", "weak", "medium"])
        ax.set_xlabel("Fault intensity")
        ax.set_title(SCEN_LABEL[f"{kind}m"].split(" (")[0]); ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel(f"Reduction of {HOP_LABEL}\nrelative to no gate (%)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "Fig2_fault_intensity.png"); fig.savefig(p, dpi=300); out.append(p)
    plt.close(fig)

    # ---- Fig. 3: expected cost ---------------------------------------------------
    ratios = np.logspace(0, 2, 60)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, scen in zip(axes, COST_PANELS):
        for g in GATES:
            r_ = T3[(T3.scen == scen) & (T3.gate == g)]
            if len(r_):
                ax.plot(ratios, ratios * r_.ahop.iloc[0] + r_.defer.iloc[0], color=GATE_COLORS[g],
                        label=g, lw=2.2 if g == G_SCSP else 1.0)
        ax.set_xscale("log"); ax.set_title(f"{scen}: {SCEN_LABEL[scen]}", fontsize=9)
        ax.set_xlabel(r"Cost ratio $c_{late}/c_{defer}$")
    axes[0].set_ylabel("Expected cost per window")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "Fig3_expected_cost.png"); fig.savefig(p, dpi=300); out.append(p)
    plt.close(fig)
    return out


# ==============================================================================
# 6. Post-hoc analyses (labelled as such in the manuscript)
#    Table 3, Table 4 Panel C, Tables 5-8, Table B.1, combined gate, AURC/AUROC,
#    break-even cost ratios
# ==============================================================================
def analysis_posthoc(run, main, save):
    S, keys = run.S, run.keys
    P95 = OP_MAIN
    out = {}

    # ---- Table 3: excess over random deferral at the same deferral rate -------------
    r95 = main["rq"][main["rq"].op == P95]
    rows = []
    for g in [x for x in GATES if x != G_NONE]:
        fam = []
        for s in RANDOM_SCEN:
            d = r95[(r95.gate == g) & (r95.scen == s)]
            ex = (d.ahop - (1.0 - d.defer) * d.late0).values
            med, p = one_sample_wilcoxon(ex)
            fam.append(dict(gate=g, scen=s, n=len(ex), median_excess=med, p=p,
                            n_negative=int((ex < 0).sum())))
        fam = pd.DataFrame(fam)
        fam["q"] = bh(fam.p.values)
        rows.append(fam)
    out["T3_random"] = save(pd.concat(rows, ignore_index=True), "Table3_random_deferral.csv")

    # ---- window-level loop over qualified combinations -------------------------------
    A, B, C, D, E, Fr, CORR, ABL, POL = [], [], [], [], [], [], [], [], []
    for c in run.qual:
        ref = run.signals(c, "CAL")
        cal_abl = {v: variant_score(ref, ref, cb) for v, cb in ABLATION.items()}
        cal_cmb = {v: variant_score(ref, ref, cb) for v, cb in COMBOS.items()}
        thr_cmb = {v: np.quantile(cal_cmb[v], P95 / 100.0) for v in COMBOS}
        # combined gate: common percentile p' (bisection) so that the union defers <= 5% of calibration
        lo_p, hi_p = 95.0, 100.0
        for _ in range(30):
            mid = 0.5 * (lo_p + hi_p)
            u = ((cal_cmb["FULL"] > np.quantile(cal_cmb["FULL"], mid / 100.0))
                 | (S["CAL__R"] > rule_bank_thresholds(S["CAL__R"], mid)).any(axis=1))
            lo_p, hi_p = (mid, hi_p) if u.mean() > COMBINED_BUDGET else (lo_p, mid)
        thr_b, rbt_b = np.quantile(cal_cmb["FULL"], hi_p / 100.0), rule_bank_thresholds(S["CAL__R"], hi_p)
        zc = np.column_stack([robust_z(ref[k], ref[k]) for k in SIGNALS])
        rho = spearmanr(zc).correlation
        CORR.append(dict(model=c[0], seed=c[1], **{f"{a}{b}": rho[i, j] for i, a in enumerate(SIGNALS)
                                                    for j, b in enumerate(SIGNALS) if i < j}))
        for key in keys:
            scen = scen_of(key)
            base = dict(model=c[0], seed=c[1], key=key, scen=scen)
            y, yp, lo = S[f"{key}__y"], run.P[c][f"{key}__yp"], run.P[c][f"{key}__lo"]
            sig = run.signals(c, key)
            late = (yp - y) > TAU
            sc_cmb = {v: variant_score(sig, ref, cb) for v, cb in COMBOS.items()}
            acc = {G_SCSP: sc_cmb["FULL"] <= thr_cmb["FULL"],
                   G_RULE: run.rule_accept(key, run.RB_THR[P95]),
                   G_P: quantile_gate(ref["P"], sig["P"], P95),
                   G_F: quantile_gate(ref["F"], sig["F"], P95),
                   G_W: quantile_gate(ref["W"], sig["W"], P95)}
            if (c, key) in run.ens:
                acc[G_ENS] = quantile_gate(run.ens[(c, "CAL")], run.ens[(c, key)], P95)
            m_s = met(acc[G_SCSP], late)

            # Table 4 Panel C: SC-SP defers as many windows (highest scores) as the comparison gate
            for g in [G_RULE, G_P, G_F, G_W, G_ENS]:
                if g in acc:
                    k = int((~acc[g]).sum()); m_g = met(acc[g], late)
                    m_m = met(topk_defer(sc_cmb["FULL"], k), late)
                    A.append(dict(**base, gate=g, defer=m_g["defer"], ahop_gate=m_g["ahop"],
                                  ahop_scsp_matched=m_m["ahop"], diff=m_m["ahop"] - m_g["ahop"],
                                  ahop_scsp_p95=m_s["ahop"]))
            # Table B.1: late threshold
            for t in TAUS:
                lt = (yp - y) > t
                r = dict(**base, tau=t, late0=float(lt.mean()))
                for g in [G_SCSP, G_RULE, G_F]:
                    mm = met(acc[g], lt)
                    r[f"ahop_{g}"], r[f"defer_{g}"], r[f"capture_{g}"] = mm["ahop"], mm["defer"], mm["capture"]
                B.append(r)
            # Table 8: signal combinations with direct matching
            for v in COMBOS:
                a = sc_cmb[v] <= thr_cmb[v]; mm = met(a, late); k = int((~a).sum())
                C.append(dict(**base, variant=v, **mm,
                              full_matched=met(topk_defer(sc_cmb["FULL"], k), late)["ahop"]))
            # combined gate
            for nm, a in {"SC-SP": acc[G_SCSP], "RB": acc[G_RULE],
                          "OR_p95": acc[G_SCSP] & acc[G_RULE],
                          "OR_budget": (sc_cmb["FULL"] <= thr_b) & run.rule_accept(key, rbt_b)}.items():
                fc = a & (yp > H_DECISION) & (y <= H_DECISION)
                D.append(dict(**base, gate=nm, **met(a, late), fcr=float(fc.mean())))
            # Table 5: lower-bound policy without deferral
            E.append(dict(**base, late_point=float(late.mean()), late_lower=float(((lo - y) > TAU).mean()),
                          ahop_scsp=m_s["ahop"], defer_scsp=m_s["defer"],
                          fcr_scsp=float((acc[G_SCSP] & (yp > H_DECISION) & (y <= H_DECISION)).mean()),
                          fcr_point=float(((yp > H_DECISION) & (y <= H_DECISION)).mean()),
                          fcr_lower=float(((lo > H_DECISION) & (y <= H_DECISION)).mean()),
                          early_point=float(((y - yp) > TAU).mean()), early_lower=float(((y - lo) > TAU).mean()),
                          under_point=float(np.maximum(0, y - yp).mean()),
                          under_lower=float(np.maximum(0, y - lo).mean())))
            # threshold-free metrics
            scores = {G_SCSP: sc_cmb["FULL"], G_F: sig["F"], G_W: sig["W"], G_P: yp}
            if (c, key) in run.ens:
                scores[G_ENS] = run.ens[(c, key)]
            for g, s_ in scores.items():
                au, ar = aurc_auroc(np.asarray(s_, float), late)
                Fr.append(dict(**base, gate=g, aurc=au, auroc=ar, late0=float(late.mean())))
            # Table 7: signal ablation (all operating points, for interpolation)
            for v, cb in ABLATION.items():
                sv = variant_score(sig, ref, cb)
                for pct in OP_PCTS:
                    ABL.append(dict(**base, variant=v, op=pct, **met(quantile_gate(cal_abl[v], sv, pct), late)))
            # Table 6: maintenance policy scenarios
            a_s = acc[G_SCSP]; d_s = ~a_s
            late_lo = (lo - y) > TAU
            fc, fc_lo = (yp > H_DECISION) & (y <= H_DECISION), (lo > H_DECISION) & (y <= H_DECISION)
            early, early_lo = (y - yp) > TAU, (y - lo) > TAU
            h0, h = late.mean(), (a_s & late).mean()
            pr = dict(**base, defer=d_s.mean(), P0_late=h0, P0_fcr=fc.mean(), P0_early=early.mean(),
                      P1_late_rho1=h, P1_late_rho08=h + 0.2 * (h0 - h), P1_late_rho05=h + 0.5 * (h0 - h),
                      P1_fcr=(a_s & fc).mean(),
                      P2a_late=((a_s & late) | (d_s & late_lo)).mean(),
                      P2a_fcr=((a_s & fc) | (d_s & fc_lo)).mean(),
                      P2a_early=((a_s & early) | (d_s & early_lo)).mean(),
                      LB_early=early_lo.mean(), LB_late=late_lo.mean(),
                      P2b_premature=(d_s & (y > H_DECISION)).mean(),
                      P2b_mean_rul_deferred=float(y[d_s].mean()) if d_s.any() else np.nan)
            for dl in (5, 10, 20):
                pr[f"P1_fail_before_review_{dl}"] = (d_s & (y <= dl)).mean()
            POL.append(pr)
        print(f"[POST-HOC] {c[0]} seed={c[1]}")

    idx = ["model", "seed", "scen"]
    avg = lambda rows, extra: pd.DataFrame(rows).groupby(idx + extra, as_index=False).mean(numeric_only=True)
    A, B, C, D, E, Fr = (avg(A, ["gate"]), avg(B, ["tau"]), avg(C, ["variant"]),
                         avg(D, ["gate"]), avg(E, []), avg(Fr, ["gate"]))
    POL = avg(POL, [])
    CORR = pd.DataFrame(CORR)
    for nm, df in dict(Table4C_matched_percombo=A, TableB1_tau_percombo=B, Table8_combos_percombo=C,
                       combined_gate_percombo=D, Table5_lower_bound_percombo=E, aurc_auroc_percombo=Fr,
                       signal_correlation_calibration=CORR, Table6_policy_percombo=POL).items():
        save(df, f"{nm}.csv")

    # ---- Table 4 Panel C tests ----------------------------------------------------
    out["T4C"] = save(family_tests(A, "diff", "gate", [g for g in [G_RULE, G_P, G_F, G_W, G_ENS]
                                                       if g in set(A.gate)], MAIN_SCEN),
                      "Table4C_matched_tests.csv")

    # ---- Table B.1 ---------------------------------------------------------------
    brows = []
    for t in TAUS:
        sub = B[B.tau == t]
        tests = {}
        for lab, col in [("vs_random", None), ("vs_RB", f"ahop_{G_RULE}"), ("vs_GF", f"ahop_{G_F}")]:
            res = []
            for s in MAIN_SCEN:
                g = sub[sub.scen == s]
                x = (g[f"ahop_{G_SCSP}"] - ((1 - g[f"defer_{G_SCSP}"]) * g.late0 if col is None else g[col])).values
                res.append(one_sample_wilcoxon(x))
            tests[lab] = (res, bh([p for _, p in res]))
        for i, s in enumerate(MAIN_SCEN):
            g = sub[sub.scen == s]
            l0, ah = g.late0.mean(), g[f"ahop_{G_SCSP}"].mean()
            de, cp = g[f"defer_{G_SCSP}"].mean(), g[f"capture_{G_SCSP}"].mean()
            row = dict(tau=t, scen=s, late0=l0, ahop=ah, defer=de,
                       reduction=(1 - ah / l0) if l0 > 0 else np.nan, lift=cp / de if de > 0 else np.nan)
            for lab, (res, q) in tests.items():
                row[f"med_{lab}"], row[f"q_{lab}"] = res[i][0], q[i]
            brows.append(row)
    out["TB1"] = save(pd.DataFrame(brows), "TableB1_tau.csv")

    # ---- Table 8 -----------------------------------------------------------------
    srows = []
    for s in MAIN_SCEN:
        for v in COMBOS:
            g = C[(C.scen == s) & (C.variant == v)]
            d_ = g.defer.mean()
            srows.append(dict(scen=s, variant=v, defer=d_, ahop=g.ahop.mean(),
                              lift=g.capture.mean() / d_ if d_ > 0 else np.nan))
    out["T8_sum"] = save(pd.DataFrame(srows), "Table8_combos_summary.csv")
    C = C.assign(diff=C.full_matched - C.ahop)
    out["T8_tests"] = save(family_tests(C, "diff", "variant", [v for v in COMBOS if v != "FULL"], MAIN_SCEN),
                           "Table8_combos_tests.csv")

    # ---- combined gate --------------------------------------------------------------
    drows = []
    for s in MAIN_SCEN:
        for g_ in ["SC-SP", "RB", "OR_p95", "OR_budget"]:
            g = D[(D.scen == s) & (D.gate == g_)]
            d_ = g.defer.mean()
            drows.append(dict(scen=s, gate=g_, defer=d_, ahop=g.ahop.mean(),
                              lift=g.capture.mean() / d_ if d_ > 0 else np.nan, fcr=g.fcr.mean()))
    out["comb"] = save(pd.DataFrame(drows), "combined_gate_summary.csv")

    # ---- Table 5 -------------------------------------------------------------------
    cols5 = ["late_point", "late_lower", "ahop_scsp", "defer_scsp", "fcr_point", "fcr_lower", "fcr_scsp",
             "early_point", "early_lower", "under_point", "under_lower"]
    out["T5"] = save(E.groupby("scen", as_index=False)[cols5].mean().set_index("scen")
                     .loc[MAIN_SCEN].reset_index(), "Table5_lower_bound.csv")

    # ---- Table 6 -------------------------------------------------------------------
    out["T6"] = save(POL.groupby("scen").mean(numeric_only=True).drop(columns=["seed"])
                     .loc[MAIN_SCEN].reset_index(), "Table6_policy.csv")

    # ---- AURC / AUROC --------------------------------------------------------------
    out["aurc"] = save(Fr.groupby(["scen", "gate"], as_index=False)[["aurc", "auroc", "late0"]].mean(),
                       "aurc_auroc_summary.csv")

    # ---- Table 7: signal ablation ---------------------------------------------------
    ABL = pd.DataFrame(ABL)
    save(ABL, "Table7_ablation_window_metrics.csv")
    chk = ABL[(ABL.variant == "FULL")].merge(
        main["RAW"][main["RAW"].gate == G_SCSP], on=["model", "seed", "key", "op"], suffixes=("", "_ref"))
    out["ablation_check"] = float(np.nanmax(np.abs(chk[["defer", "ahop"]].values
                                                    - chk[["defer_ref", "ahop_ref"]].values)))
    if out["ablation_check"] > 1e-9:
        raise RuntimeError("FULL ablation variant differs from SC-SP; do not use the results.")
    p95 = (ABL[ABL.op == P95].groupby(["model", "seed", "scen", "variant"], as_index=False)
           .mean(numeric_only=True))
    SUM = p95.groupby(["scen", "variant"], as_index=False)[["defer", "ahop", "late0", "capture"]].mean()
    SUM["lift"] = np.where(SUM.defer > 0, SUM.capture / SUM.defer.clip(lower=1e-12), np.nan)
    SUM["reduction"] = 1.0 - SUM.ahop / SUM.late0
    out["T7_sum"] = save(SUM, "Table7_ablation_summary.csv")

    def curve(g):
        l0 = float(g.late0.iloc[0])
        pts = pd.DataFrame([(0.0, l0), (1.0, 0.0)] + list(zip(g.defer, g.ahop)), columns=["x", "y"])
        pts = pts.groupby("x", as_index=False)["y"].min()
        return pts.x.values, pts.y.values

    mrows = []
    for (m, s, key, scen), g in ABL[ABL.scen.isin(MAIN_SCEN)].groupby(["model", "seed", "key", "scen"]):
        xs, ys = curve(g[g.variant == "FULL"])
        full95 = g[(g.variant == "FULL") & (g.op == P95)].iloc[0]
        for v in ABLATION:
            if v == "FULL":
                continue
            r95_ = g[(g.variant == v) & (g.op == P95)].iloc[0]
            mrows.append(dict(model=m, seed=s, scen=scen, variant=v,
                              diff_p95=float(full95.ahop - r95_.ahop),
                              diff_matched=float(np.interp(r95_.defer, xs, ys) - r95_.ahop)))
    M = pd.DataFrame(mrows).groupby(["model", "seed", "scen", "variant"], as_index=False).mean(numeric_only=True)
    t7 = []
    for v in [k for k in ABLATION if k != "FULL"]:
        fam = []
        for sc in MAIN_SCEN:
            g = M[(M.variant == v) & (M.scen == sc)]
            m95, p95v = one_sample_wilcoxon(g.diff_p95.values)
            mm, pm = one_sample_wilcoxon(g.diff_matched.values)
            fam.append(dict(variant=v, scen=sc, n=len(g), med_p95=m95, p_p95=p95v, med_matched=mm, p_matched=pm))
        fam = pd.DataFrame(fam)
        fam["q_p95"], fam["q_matched"] = bh(fam.p_p95.values), bh(fam.p_matched.values)
        t7.append(fam)
    out["T7"] = save(pd.concat(t7, ignore_index=True), "Table7_ablation_tests.csv")

    # ---- break-even cost ratios (Section 5.5) ----------------------------------------
    T3 = main["T3"]
    brk = []
    for s in MAIN_SCEN:
        g0 = T3[(T3.scen == s) & (T3.gate == G_NONE)].iloc[0]
        gs = T3[(T3.scen == s) & (T3.gate == G_SCSP)].iloc[0]
        gf = T3[(T3.scen == s) & (T3.gate == G_F)].iloc[0]
        dh = g0.ahop - gs.ahop
        cross = ((gf.defer - gs.defer) / (gs.ahop - gf.ahop)) if gs.ahop > gf.ahop else np.nan
        brk.append(dict(scen=s, defer_scsp=gs.defer, h0=g0.ahop, h_scsp=gs.ahop,
                        break_even=gs.defer / dh if dh > 0 else np.nan,
                        break_even_rho08=gs.defer / (0.8 * dh) if dh > 0 else np.nan,
                        gf_cheaper_above=cross))
    out["breakeven"] = save(pd.DataFrame(brk), "break_even_cost_ratios.csv")
    out["corr"] = CORR
    return out


def family_tests(df, col, by, cats, scens):
    """Median of a per-combination difference, Wilcoxon, BH within each category over scenarios."""
    rows = []
    for g in cats:
        fam = []
        for s in scens:
            x = df[(df[by] == g) & (df.scen == s)][col].values
            med, p = one_sample_wilcoxon(x)
            fam.append({by: g, "scen": s, "n": int(np.isfinite(x).sum()), "median": med, "p": p})
        fam = pd.DataFrame(fam)
        fam["q"] = bh(fam.p.values)
        rows.append(fam)
    return pd.concat(rows, ignore_index=True)


# ==============================================================================
# 7. Human-readable tables, frozen-run check, manifest
# ==============================================================================
def write_markdown(main, post, path):
    pc = lambda x, d=2: "–" if not np.isfinite(x) else f"{x * 100:.{d}f}"
    pp = lambda x: "–" if not np.isfinite(x) else f"{x * 100:+.2f}"
    st = lambda q: "*" if np.isfinite(q) and q < 0.05 else ""
    L = [f"# SC-SP results — {RUN_TAG} (split seed {SPLIT_SEED})", "",
         f"Qualified combinations: {main['verdict']['n_qualified']} of {main['verdict']['n_combos']}. "
         f"Rates in %, differences in pp, * = BH q < 0.05.", ""]

    L += ["## Table 1. Scenarios", "", "| ID | Scenario | Windows | Units | Out-of-range | Late rate |",
          "|---|---|---|---|---|---|"]
    for r in main["T1"].itertuples():
        L.append(f"| {r.scen} | {r.label} | {r.n:,.0f} | {r.n_units:.0f} | {pc(r.oob_frac)} | {pc(r.base_late, 1)} |")

    L += ["", "## Table 2. Predictors (RMSE mean ± sd, late, coverage, MPIW)", "",
          "| Model | Scen. | Seeds | RMSE | Late | Coverage | MPIW |", "|---|---|---|---|---|---|---|"]
    for r in main["T2"].itertuples():
        L.append(f"| {r.model} | {r.scen} | {r.n_seeds} | {r.rmse:.1f} ± {r.rmse_sd:.1f} | {pc(r.late, 1)} | "
                 f"{r.coverage:.2f} | {r.mpiw:.1f} |")

    T = post["T3_random"]
    gates = [g for g in GATES if g != G_NONE]
    L += ["", "## Table 3. Excess over random deferral (median, pp)", "",
          "| Scen. | " + " | ".join(gates) + " |", "|---" * (len(gates) + 1) + "|"]
    for s in RANDOM_SCEN:
        cells = []
        for g in gates:
            r = T[(T.gate == g) & (T.scen == s)].iloc[0]
            cells.append(f"{pp(r.median_excess)}{st(r.q)} ({r.n_negative}/{r.n} neg.)")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    T4 = main["T4"]
    L += ["", "## Table 4 Panel A (pre-specified)", "",
          "| Hyp. | Scen. | Contrast | Median | Engine-level mean [95% CI] | q | Supported |",
          "|---|---|---|---|---|---|---|"]
    for r in T4[T4.family.str.startswith("B-H")].itertuples():
        L.append(f"| {r.family} | {r.scen} | {r.gate} − {r.vs} | {pp(r.median_diff)} | "
                 f"{pp(r.unit_mean_diff)} [{pp(r.unit_lo95)}, {pp(r.unit_hi95)}] | {r.q:.2g} | "
                 f"{'yes' if r.supports else 'no'} |")
    vs = [f"vs {g}" for g in (G_P, G_F, G_W, G_ENS)]
    L += ["", "## Table 4 Panel B (SC-SP − gate at p = 95)", "",
          "| Scen. | " + " | ".join("−" + v[3:] for v in vs) + " |", "|---" * (len(vs) + 1) + "|"]
    for s in MAIN_SCEN:
        cells = []
        for v in vs:
            r = T4[(T4.family == v) & (T4.scen == s)].iloc[0]
            cells.append(f"{pp(r.median_diff)}{st(r.q)}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    def fam_md(df, by, cats, title, head=None):
        L.extend(["", title, "", "| Scen. | " + " | ".join(head or cats) + " |", "|---" * (len(cats) + 1) + "|"])
        for s in MAIN_SCEN:
            cells = []
            for g in cats:
                r = df[(df[by] == g) & (df.scen == s)].iloc[0]
                cells.append(f"{pp(r['median'])}{st(r.q)}")
            L.append(f"| {s} | " + " | ".join(cells) + " |")

    A = post["T4C"]
    fam_md(A, "gate", [g for g in [G_RULE, G_P, G_F, G_W, G_ENS] if g in set(A.gate)],
           "## Table 4 Panel C (matched deferral, SC-SP − gate)")

    T5 = post["T5"]
    L += ["", "## Table 5. Lower-bound policy without deferral", "",
          "| Scen. | Late point | Late lower | SC-SP (defer.) | FCR point | FCR lower | FCR SC-SP | "
          "Early point | Early lower | Underpred. point | Underpred. lower |", "|---" * 11 + "|"]
    for r in T5.itertuples():
        L.append(f"| {r.scen} | {pc(r.late_point)} | {pc(r.late_lower)} | {pc(r.ahop_scsp)} ({pc(r.defer_scsp, 1)}) | "
                 f"{pc(r.fcr_point)} | {pc(r.fcr_lower)} | {pc(r.fcr_scsp)} | {pc(r.early_point, 1)} | "
                 f"{pc(r.early_lower, 1)} | {r.under_point:.1f} | {r.under_lower:.1f} |")

    T6 = post["T6"]
    L += ["", "## Table 6. Maintenance policy scenarios", "",
          "| Scen. | Defer. | P0 late | P1 ρ=1 | ρ=0.8 | ρ=0.5 | Δ=10 | Δ=20 | P2a late | P2a FCR | "
          "P2a early | LB all early | P2b r>30 | Mean r |", "|---" * 14 + "|"]
    for r in T6.itertuples():
        L.append(f"| {r.scen} | {pc(r.defer, 1)} | {pc(r.P0_late)} | {pc(r.P1_late_rho1)} | {pc(r.P1_late_rho08)} | "
                 f"{pc(r.P1_late_rho05)} | {pc(r.P1_fail_before_review_10)} | {pc(r.P1_fail_before_review_20)} | "
                 f"{pc(r.P2a_late)} | {pc(r.P2a_fcr)} | {pc(r.P2a_early, 1)} | {pc(r.LB_early, 1)} | "
                 f"{pc(r.P2b_premature, 1)} | {r.P2b_mean_rul_deferred:.1f} |")

    T7 = post["T7"]
    L += ["", "## Table 7. Signal ablation (SC-SP interpolated at the variant's p = 95 deferral − variant)", "",
          "| Scen. | " + " | ".join(ABLATION_REPORT) + " |", "|---" * (len(ABLATION_REPORT) + 1) + "|"]
    for s in MAIN_SCEN:
        cells = []
        for v in ABLATION_REPORT:
            r = T7[(T7.variant == v) & (T7.scen == s)].iloc[0]
            cells.append(f"{pp(r.med_matched)}{st(r.q_matched)}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    S7 = post["T7_sum"]
    L += ["", "Ablation p = 95 deferral (%) / Aut. HOP@20 (%) / lift:", "",
          "| Scen. | " + " | ".join(ABLATION) + " |", "|---" * (len(ABLATION) + 1) + "|"]
    for s in MAIN_SCEN:
        cells = []
        for v in ABLATION:
            r = S7[(S7.scen == s) & (S7.variant == v)].iloc[0]
            cells.append(f"{pc(r.defer, 1)} / {pc(r.ahop)} / {r.lift:.2f}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    S8, T8 = post["T8_sum"], post["T8_tests"]
    vs8 = [v for v in COMBOS if v != "FULL"]
    L += ["", "## Table 8. Signal combinations", "",
          "| Scen. | " + " | ".join(["SC-SP"] + vs8) + " | " + " | ".join(f"SC-SP − {v}" for v in vs8) + " |",
          "|---" * (1 + len(COMBOS) + len(vs8)) + "|"]
    for s in MAIN_SCEN:
        left = []
        for v in COMBOS:
            r = S8[(S8.scen == s) & (S8.variant == v)].iloc[0]
            left.append(f"{pc(r.defer)}/{pc(r.ahop)}/{r.lift:.2f}")
        right = []
        for v in vs8:
            r = T8[(T8.variant == v) & (T8.scen == s)].iloc[0]
            right.append(f"{pp(r['median'])}{st(r.q)}")
        L.append(f"| {s} | " + " | ".join(left + right) + " |")
    cr = post["corr"]
    L += ["", "Spearman correlation of the signals on the calibration split (mean [min, max]): " +
          "; ".join(f"{k} {cr[k].mean():+.2f} [{cr[k].min():+.2f}, {cr[k].max():+.2f}]"
                    for k in cr.columns if k not in ("model", "seed"))]

    CB = post["comb"]
    L += ["", "## Combined gate (Section 5.3): deferral / Aut. HOP@20 / lift / FCR@30", "",
          "| Scen. | SC-SP | RB | OR p95 | OR 5% budget |", "|---|---|---|---|---|"]
    for s in MAIN_SCEN:
        cells = []
        for g in ["SC-SP", "RB", "OR_p95", "OR_budget"]:
            r = CB[(CB.scen == s) & (CB.gate == g)].iloc[0]
            cells.append(f"{pc(r.defer, 1)} / {pc(r.ahop)} / {r.lift:.2f} / {pc(r.fcr)}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    AU = post["aurc"]
    ga = [g for g in [G_SCSP, G_F, G_W, G_P, G_ENS] if g in set(AU.gate)]
    L += ["", "## Threshold-free metrics (Section 5.4): AURC (%) / AUROC", "",
          "| Scen. | " + " | ".join(ga) + " | late0 |", "|---" * (len(ga) + 2) + "|"]
    for s in MAIN_SCEN:
        g_ = AU[AU.scen == s]
        L.append(f"| {s} | " + " | ".join(
            f"{pc(g_[g_.gate == g].aurc.iloc[0])} / {g_[g_.gate == g].auroc.iloc[0]:.3f}" for g in ga)
            + f" | {pc(g_.late0.mean())} |")

    BE = post["breakeven"]
    L += ["", "## Break-even cost ratios (Section 5.5)", "",
          "| Scen. | d/(h0−h) | ρ = 0.8 | GF cheaper above |", "|---|---|---|---|"]
    for r in BE.itertuples():
        f1 = lambda v: "–" if not np.isfinite(v) else f"{v:.1f}"
        L.append(f"| {r.scen} | {f1(r.break_even)} | {f1(r.break_even_rho08)} | {f1(r.gf_cheaper_above)} |")

    TB = post["TB1"]
    L += ["", "## Table B.1. Late threshold: late0 / reduction / lift (SC-SP − random, − RB, − GF)", "",
          "| τ | Scen. | late0 | reduction | lift | vs random | vs RB | vs GF |", "|---" * 8 + "|"]
    for r in TB.itertuples():
        L.append(f"| {r.tau:.0f} | {r.scen} | {pc(r.late0, 1)} | {pc(r.reduction, 0)} | {r.lift:.2f} | "
                 f"{pp(r.med_vs_random)}{st(r.q_vs_random)} | {pp(r.med_vs_RB)}{st(r.q_vs_RB)} | "
                 f"{pp(r.med_vs_GF)}{st(r.q_vs_GF)} |")

    T3 = main["T3"]
    L += ["", "## Table A.1. Results per gate at p = 95", "",
          "| Scen. | Gate | Deferral | Aut. HOP@20 | Reduction | Lift | FCR@30 |", "|---|---|---|---|---|---|---|"]
    for r in T3.itertuples():
        L.append(f"| {r.scen} | {r.gate} | {pc(r.defer, 1)} | {pc(r.ahop)} | {pc(r.reduction, 1)} | "
                 f"{'–' if not np.isfinite(r.lift) else f'{r.lift:.2f}'} | {pc(r.fcr)} |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def compare_frozen(new, frozen_path, keys, cols):
    """Max absolute difference between a regenerated table and a frozen CSV of the earlier scripts."""
    if not os.path.exists(frozen_path):
        return None
    old = pd.read_csv(frozen_path)
    new = new.copy()
    for k in keys:
        new[k] = new[k].astype(str); old[k] = old[k].astype(str)
    mrg = new.merge(old, on=keys, suffixes=("", "_old"))
    cols = [c for c in cols if c in mrg and f"{c}_old" in mrg]
    if not len(mrg) or not cols:
        return dict(n=int(len(mrg)), max_abs_diff=None)
    d = np.abs(mrg[cols].values.astype(float) - mrg[[f"{c}_old" for c in cols]].values.astype(float))
    return dict(n=int(len(mrg)), max_abs_diff=float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)


def frozen_checks(main, post):
    fp, fr, fa = FROZEN["paper"], FROZEN["revision"], FROZEN["ablation"]
    chk = {
        "T1_scenarios": compare_frozen(main["T1"], os.path.join(fp, "T1_scenarios.csv"), ["scen"],
                                       ["n", "n_units", "oob_frac", "base_late"]),
        "T2_predictors": compare_frozen(main["T2"], os.path.join(fp, "T2_predictors.csv"), ["model", "scen"],
                                        ["rmse", "rmse_sd", "late", "coverage", "mpiw"]),
        "T3_gates(Table A.1)": compare_frozen(main["T3"], os.path.join(fp, "T3_gates.csv"), ["scen", "gate"],
                                              ["defer", "ahop", "reduction", "lift", "fcr", "fcr_acc"]),
        "T4_tests": compare_frozen(main["T4"], os.path.join(fp, "T4_tests.csv"), ["family", "scen", "gate", "vs"],
                                   ["median_diff", "p", "q", "unit_mean_diff", "unit_lo95", "unit_hi95"]),
        "ablation_tests(Table 7)": compare_frozen(post["T7"], os.path.join(fa, "ablation_tests.csv"),
                                                  ["variant", "scen"], ["med_p95", "p_p95", "med_matched", "p_matched"]),
        "ablation_full_vs_scsp": dict(max_abs_diff=post["ablation_check"]),
    }
    for nm, keys, cols in [("A_matched_exact", ["model", "seed", "scen", "gate"], ["diff", "ahop_gate"]),
                           ("B_tau", ["model", "seed", "scen", "tau"], ["late0", "ahop_SC-SP"]),
                           ("C_variants", ["model", "seed", "scen", "variant"], ["defer", "ahop", "full_matched"]),
                           ("D_or_gate", ["model", "seed", "scen", "gate"], ["defer", "ahop", "fcr"]),
                           ("E_lower_policy", ["model", "seed", "scen"], ["late_lower", "early_lower", "under_lower"]),
                           ("F_aurc", ["model", "seed", "scen", "gate"], ["aurc", "auroc"]),
                           ("G_policy", ["model", "seed", "scen"], ["P1_late_rho05", "P2a_late", "P2b_premature"])]:
        new_name = {"A_matched_exact": "Table4C_matched_percombo", "B_tau": "TableB1_tau_percombo",
                    "C_variants": "Table8_combos_percombo", "D_or_gate": "combined_gate_percombo",
                    "E_lower_policy": "Table5_lower_bound_percombo", "F_aurc": "aurc_auroc_percombo",
                    "G_policy": "Table6_policy_percombo"}[nm]
        new = pd.read_csv(os.path.join(TAB_DIR, f"{new_name}.csv"))
        if "tau" in keys:
            new["tau"] = new["tau"].astype(float)
        chk[nm] = compare_frozen(new, os.path.join(fr, f"{nm}_percombo.csv"), keys, cols)
    return chk


def write_manifest(outputs, versions):
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    path = globals().get("__file__")
    man = dict(
        script=SCRIPT_NAME, script_sha=sha(os.path.abspath(path)) if path and os.path.exists(path) else None,
        anchor_doc=ANCHOR_DOC, run_tag=RUN_TAG, stage=STAGE, utc=datetime.now(timezone.utc).isoformat(),
        fast_smoke=FAST_SMOKE,
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      **versions),
        config=dict(seeds=SEEDS, models=MODEL_NAMES, ensemble_groups=ENSEMBLE_GROUPS, split_seed=SPLIT_SEED,
                    eval_win=EVAL_WIN, tau=TAU, taus=TAUS, h_decision=H_DECISION, alpha_cqr=ALPHA_CQR,
                    op_pcts=OP_PCTS, op_main=OP_MAIN, qual_rmse_max=QUAL_RMSE_MAX,
                    qual_std_ratio=QUAL_PRED_STD_RATIO, shrink_f=SHRINK_F, shrink_h=SHRINK_H,
                    n_real=N_REAL, fault_k=FAULT_K, fault_onset=FAULT_ONSET, fault_levels=FAULT_LEVELS,
                    fault_sensors_fixed=FAULT_SENSORS_FIXED, base_late_min=BASE_LATE_MIN,
                    s0_defer_max=S0_DEFER_MAX, h2_min_support=H2_MIN_SUPPORT, n_boot=N_BOOT,
                    boot_seed=BOOT_SEED, combined_budget=COMBINED_BUDGET),
        outputs={os.path.relpath(p, OUT_DIR): sha(p) for p in outputs})
    p = os.path.join(OUT_DIR, "run_manifest.json")
    json.dump(man, open(p, "w"), indent=2, default=str)
    print(f"[MANIFEST] {p}")


def stage_analyze(versions):
    os.makedirs(TAB_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    outputs = []

    def save(df, name):
        p = os.path.join(TAB_DIR, name)
        df.to_csv(p, index=False)
        outputs.append(p)
        return df

    run = Run()
    print(f"[ANALYZE] {RUN_DIR}: {len(run.combos)} combinations, {len(run.qual)} qualified, {len(run.keys)} keys")
    main = analysis_main(run, save)
    outputs += main["figures"]
    post = analysis_posthoc(run, main, save)

    info_path = os.path.join(RUN_DIR, "compute_info.json")
    old_man = os.path.join(RUN_DIR, "run_manifest.json")
    if os.path.exists(info_path):
        main["verdict"]["compute_info"] = json.load(open(info_path))
    elif os.path.exists(old_man):                   # frozen run20260923_v38
        cfg = json.load(open(old_man)).get("config", {})
        main["verdict"]["compute_info"] = dict(split_seed=cfg.get("split_seed"),
                                               fault_sensors=cfg.get("fault_sensors"))
    p = os.path.join(OUT_DIR, "verdicts.json")
    json.dump(main["verdict"], open(p, "w"), indent=2, ensure_ascii=False, default=float)
    outputs.append(p)

    md = os.path.join(OUT_DIR, "PAPER_TABLES.md")
    write_markdown(main, post, md)
    outputs.append(md)

    chk = frozen_checks(main, post)
    p = os.path.join(OUT_DIR, "frozen_check.json")
    json.dump(chk, open(p, "w"), indent=2, default=float)
    outputs.append(p)
    bad = {k: v for k, v in chk.items() if v and v.get("max_abs_diff") not in (None,) and v["max_abs_diff"] > 1e-9}
    print("[FROZEN CHECK]", json.dumps(chk, indent=1, default=float))
    if bad:
        print("[WARN] differs from the frozen outputs:", list(bad))
    print(json.dumps(main["verdict"], indent=2, ensure_ascii=False, default=float))
    write_manifest(outputs, versions)
    print(f"[OUT] {OUT_DIR}")


# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"{SCRIPT_NAME}  stage={STAGE}  run={RUN_DIR}  split_seed={SPLIT_SEED}")
    print("=" * 80)
    versions = {}
    if STAGE in ("all", "compute"):
        versions.update(stage_compute())
    if STAGE in ("all", "analyze"):
        stage_analyze(versions)
