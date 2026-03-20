# ⚡ SENTRY — Sensor-based ENhanced Temporal analYsis Framework

> **Anomalies are not loud numbers — they are sudden behavioural changes.**  
> SENTRY is built around detecting the *change*, not the value.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![AUC](https://img.shields.io/badge/AUC-0.9928-brightgreen)]()
[![F1](https://img.shields.io/badge/F1--anomaly-0.52-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Core Insight](#core-insight)
- [Results](#results)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Running on Kaggle](#running-on-kaggle)
- [Running Locally](#running-locally)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Validation Strategy](#validation-strategy)
- [Known Limitations](#known-limitations)

---

## Overview

SENTRY is a research-grade machine learning framework for detecting anomalies in industrial sensor data. Built for an energy manufacturing plant operating 5 continuous sensors (X1–X5), the system predicts whether a given timestep represents **normal operation (0)** or an **anomaly (1)**.

This is not a standard tabular classification problem. SENTRY treats it as a **temporal behaviour detection** problem — and that distinction drives every design decision.

---

## The Problem

| Challenge | Why It's Hard | SENTRY's Solution |
|---|---|---|
| **115:1 class imbalance** | 99.1% normal, 0.9% anomaly — accuracy is useless | F1-anomaly metric + `scale_pos_weight` + threshold tuning |
| **Time dependency** | A value of X1=250 means nothing without history | 100+ temporal features (lag, delta, z-score, EWM) |
| **Subtle anomaly patterns** | Most anomalies don't breach absolute thresholds | Variance shifts and rate-of-change features as primary signals |
| **Distribution shift** | Models degrade over time in production | Drift simulation + walk-forward CV + retraining strategy |
| **Data leakage risk** | Random splits contaminate time-series evaluation | Strict chronological split — training never sees future data |

---

## Core Insight

> *"A sensor reading of X1=250 is not inherently anomalous — but a jump from X1=80 → 250 in three timesteps almost certainly is."*

**Three proven hypotheses:**
- **H1 — Variance Instability:** Rolling std spikes precede anomalies more reliably than absolute values
- **H2 — Delta Dominance:** Rate-of-change (Δ) features carry more signal than lag features alone
- **H3 — Cross-Sensor Coupling:** Simultaneous deviation across multiple sensors is a stronger anomaly signal

All three are validated by the ablation study in the notebook.

---

## Results

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.9928 |
| **F1-anomaly (validation)** | 0.4934 |
| **F1-anomaly (5-fold CV mean)** | 0.52 ± 0.035 |
| **Recall-anomaly** | 0.5756 |
| **F1-weighted** | 0.9955 |
| **Best Model** | XGBoost [t=0.94] |
| **Imbalance ratio** | 115:1 |
| **Training samples** | 1,639,424 |

**Walk-forward Cross-Validation (5 folds):**
```
Fold 1  F1_anom=0.5115  AUC=0.9810
Fold 2  F1_anom=0.5691  AUC=0.9894
Fold 3  F1_anom=0.5158  AUC=0.9861
Fold 4  F1_anom=0.5325  AUC=0.9883
Fold 5  F1_anom=0.4711  AUC=0.9903
Mean    F1_anom=0.5200 ± 0.0355
```

---

## Project Structure

```
sentry-anomaly-detection/
│
├── notebook.ipynb              # Main notebook — full pipeline
├── submission.csv              # Final predictions on test set
├── submission.parquet          # Same predictions in parquet format
├── model.pkl                   # Saved XGBoost model
├── scaler.pkl                  # Saved RobustScaler
├── threshold.png               # Threshold optimisation plot
│
├── data/                       # Place your data files here
│   ├── train.parquet
│   └── test.parquet
│
└── README.md
```

---

## How It Works

### SENTRY Pipeline

```
RAW DATA → FEATURE ENGINE → MODELLING → CALIBRATION → DEPLOYMENT
────────   ───────────────  ──────────  ────────────  ───────────
Parquet    A. Time feats    ① RF        Threshold     Risk Score
5 sensors  B. Lag (1–10)    ② XGBoost   sweep         0.0 – 1.0
X1–X5      C. Rolling stats ③ LightGBM  0.05–0.95
Timestamped D. Delta/rate   ④ Ensemble  Optimise      NORMAL  <0.30
           E. Z-score                   F1-weighted   WATCH   0.30+
           F. EWM                                     WARN    0.60+
           G. Interaction   100+ feats                PAGE    0.85+
           H. System state
```

### Three-Tier Alert System

Rather than binary 0/1 output, SENTRY produces a continuous **anomaly risk score**:

| Score | Label | Action |
|---|---|---|
| 0.00 – 0.30 | 🟢 NORMAL | Log only |
| 0.30 – 0.60 | 🟡 WATCH | Dashboard alert |
| 0.60 – 0.85 | 🟠 WARNING | Notify supervisor |
| 0.85 – 1.00 | 🔴 CRITICAL | Page engineer immediately |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
git clone https://github.com/yourusername/sentry-anomaly-detection.git
cd sentry-anomaly-detection

pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn joblib pyarrow jupyter
```

Or install all at once:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
shap
imbalanced-learn
joblib
pyarrow
jupyter
```

---

## Running on Kaggle

This notebook is designed to run on Kaggle with the dataset attached.

### Step 1 — Upload the dataset

1. Go to [kaggle.com](https://kaggle.com) → your profile → **Datasets** → **New Dataset**
2. Upload `train.parquet` and `test.parquet`
3. Give it a title (e.g. `sensor-anomaly-data`) and click **Create**

### Step 2 — Attach dataset to notebook

1. Open your notebook on Kaggle
2. Click **+ Add Data** in the right sidebar
3. Go to **Your Datasets** → find your dataset → click **Add**

### Step 3 — Verify file paths

Run this in a cell to confirm paths:

```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

### Step 4 — Update the loading cell

Replace the data loading paths with what the above printed:

```python
train_raw = pd.read_parquet('/kaggle/input/sensor-anomaly-data/train.parquet')
test_raw  = pd.read_parquet('/kaggle/input/sensor-anomaly-data/test.parquet')
```

### Step 5 — Run All

Click **Run All** and wait (~10–15 minutes for full execution).

### Step 6 — Commit

Click **Save Version → Save & Run All (Commit)** to save the executed notebook with all outputs visible.

---

## Running Locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/tanishipss/sentry-anomaly-detection.git
cd sentry-anomaly-detection
```

### Step 2 — Place data files

```bash
mkdir data
# Copy your train.parquet and test.parquet into the data/ folder
```

### Step 3 — Update data paths in notebook

In the data loading cell, change:

```python
# From (Kaggle path):
train_raw = pd.read_parquet('/kaggle/input/.../train.parquet')

# To (local path):
train_raw = pd.read_parquet('data/train.parquet')
test_raw  = pd.read_parquet('data/test.parquet')
```

### Step 4 — Launch Jupyter

```bash
jupyter notebook
```

Open `notebook.ipynb` and click **Run All**.

### Step 5 — Using the saved model

After running the notebook once, use the saved model for predictions:

```python
import joblib
import pandas as pd
from feature_engine import build_features, FEAT  # exported from notebook

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new sensor data
df = pd.read_parquet('new_data.parquet')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Build features and predict
df_fe = build_features(df)
X_new = scaler.transform(df_fe[FEAT].fillna(0))
risk_scores = model.predict_proba(X_new)[:, 1]

print(f'Anomalies detected: {(risk_scores > 0.94).sum()}')
```

---

## Feature Engineering

SENTRY builds **100+ features** across 8 temporal families:

| Family | Features | Captures |
|---|---|---|
| **A. Time** | `hour_sin`, `hour_cos`, `is_weekend` | Cyclical time patterns |
| **B. Lags** | `X*_lag1` through `X*_lag10` | Recent sensor history |
| **C. Rolling stats** | `X*_rmean6`, `X*_rstd12`, `X*_rrange` | Local baseline behaviour |
| **D. Delta / Rate** | `X*_diff1`, `X*_accel`, `X*_absdiff` | Speed of change |
| **E. Z-score** | `X*_zscore` | Deviation from local normal |
| **F. EWM** | `X*_ewm`, `X*_ewm_dev` | Exponential trend departure |
| **G. Interactions** | `X1_X2_ratio`, cross-sensor diffs | Sensor coupling breakdown |
| **H. System state** | `n_sensors_above_2sigma` | Global alarm signal |

**Why temporal features matter (ablation-proven):**
- Raw sensors only → baseline F1
- + Temporal features → significant F1 gain
- + Model tuning → further improvement
- + Ensemble → best result

---

## Model Architecture

Three models trained in deliberate progression:

| Stage | Model | Purpose |
|---|---|---|
| 🥉 Baseline | Random Forest | Establishes non-linear baseline |
| 🥇 Primary | XGBoost | Sequential error correction + imbalance weighting |
| 🥈 Secondary | LightGBM | Confirms boosting superiority |
| 🏆 Final | Weighted Ensemble | RF + XGBoost + LightGBM, AUC-weighted voting |

**Why XGBoost wins on this data:**
- Handles 115:1 imbalance via `scale_pos_weight`
- Sequential boosting corrects errors on rare anomaly class
- Non-linear splits capture `X*_zscore × lag` interactions that linear models cannot

---

## Validation Strategy

Three independent validation layers ensure results are not lucky:

1. **Chronological 80/20 split** — training data is always earlier than validation data. No shuffling.
2. **5-fold walk-forward CV** — 5 non-overlapping future windows. Mean F1 = 0.52 ± 0.035.
3. **Drift simulation** — model trained on first 50% evaluated on 5 progressive later windows.

> Reported metrics are conservative estimates. In production the model would be retrained on more recent data before deployment.

---

## Known Limitations

| Limitation | Description | Mitigation |
|---|---|---|
| Gradual drift | Slow deterioration over 100+ steps is missed | EWM features provide partial coverage |
| Novel anomaly types | Unseen failure modes not in training data | Pair with Isolation Forest as unsupervised backup |
| Threshold sensitivity | Optimal threshold tuned on one validation window | Recalibrate periodically with fresh labelled data |
| Static feature engineering | Manually coded temporal patterns | LSTM/Transformer could learn long-range patterns |

---

## Citation

If you use SENTRY in your work:

```
SENTRY — Sensor-based ENhanced Temporal analYsis Framework
Industrial Anomaly Detection, 2024
github.com/yourusername/sentry-anomaly-detection
```

---

*Built with the principle that anomaly detection is not about identifying rare events — it is about capturing deviations in system behaviour over time.*
