# DataBattle 2026 — Lightning Storm End Prediction

> **Challenge:** Predict whether a cloud-to-ground (CG) lightning strike is the last one before 30 minutes of silence, enabling airports to reissue apron access 5–15 minutes earlier than the current conservative rule.

**Model:** LightGBM · **AUC:** 0.981 · **F1:** 0.797 · **Brier:** 0.031 · **CO₂:** 0.34 g total training

---

## Quick Start

### Option A — Local (recommended for jury)

```bash
git clone <repo-url>
cd data-Battle-2026
make app          # installs deps automatically, then opens the dashboard
```

Open [http://localhost:8501](http://localhost:8501)

### Option B — Docker (zero setup)

```bash
make app-build    # build the image (outputs baked in)
make app-docker   # run at localhost:8501
```

---

## Dashboard Commands

| Command | What it does |
|---|---|
| `make app` | Install deps if missing, start Streamlit dashboard locally |
| `make app-build` | Build Docker image with all outputs baked in |
| `make app-docker` | Run the Docker image at `localhost:8501` |
| `make app-stop` | Stop the running Docker container |

---

## Reproduce the Results

> **Prerequisite:** place `segment_alerts_all_airports_train.csv` in `data/`

```bash
make install      # create .venv and install all dependencies
make train        # feature engineering + LightGBM 5-fold CV + temporal test
make compare      # LR vs XGBoost vs LightGBM comparison (outputs/saves/)
make run-shap     # SHAP global + per-airport + waterfall figures
make predict      # generate outputs/submissions/predictions.csv
```

Or run everything in one shot:

```bash
make pipeline     # train → evaluate → predict
```

---

## All Make Targets

### Pipeline

| Command | What it does |
|---|---|
| `make pipeline` | Full run: train → evaluate → predict |
| `make train` | Feature engineering + LightGBM GroupKFold CV training |
| `make evaluate` | Print OOF metrics + gain/risk threshold sweep |
| `make predict` | Generate `outputs/submissions/predictions.csv` |
| `make compare` | Run LR vs XGBoost vs LightGBM comparison |
| `make tune` | Optuna hyperparameter search (50 trials) |

### Analysis & Figures

| Command | What it does |
|---|---|
| `make run-shap` | SHAP explainability figures → `outputs/figures/shap/` |
| `make run-compare` | Model comparison figures → `outputs/figures/model-comparison/` |
| `make run-final-eda` | EDA + feature signal figures → `outputs/figures/` |
| `make run FILE=src/train.py` | Run any script directly |

### Setup & Utilities

| Command | What it does |
|---|---|
| `make install` | Create `.venv` and install `requirements.txt` |
| `make check-all` | Syntax-check all `.py` files in `src/`, `app/` |
| `make clean` | Remove generated figures and `.ipynb` files |

---

## Project Structure

```
data-Battle-2026/
├── app/                    # Streamlit dashboard
│   ├── Home.py             # Landing page (objective, results, impact)
│   ├── pages/
│   │   ├── 1_EDA_Features.py
│   │   ├── 2_Model_Comparison.py
│   │   └── 3_Predict_Explain.py
│   └── utils/loaders.py
├── src/                    # Production source code
│   ├── features.py         # 36-feature engineering pipeline
│   ├── train.py            # LightGBM GroupKFold CV + temporal test
│   ├── compare_models.py   # LR vs XGBoost vs LightGBM benchmark
│   ├── evaluate.py         # OOF metrics + threshold sweep
│   ├── predict.py          # Inference on test data
│   └── tune.py             # Optuna hyperparameter search
├── report/
│   ├── slides.tex/.pdf     # Beamer presentation (19 slides)
│   ├── eda_report.tex/.pdf # Full EDA report
│   ├── manual.tex/.pdf     # Technical manual
│   └── figures/            # All plots used in reports
├── outputs/                # Generated at runtime (not in git)
│   ├── models/             # Trained .pkl files
│   ├── figures/            # EDA, SHAP, model-comparison plots
│   ├── saves/              # CV scores, OOF predictions, thresholds
│   ├── logs/               # CodeCarbon CO₂ tracking (carbon_report.csv)
│   └── submissions/        # predictions.csv
├── data/                   # Training data (not in git — add manually)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Key Technical Decisions

| Choice | Reason |
|---|---|
| **LightGBM** over XGBoost | +2.3 pp F1, lower CO₂ (0.099 g vs 0.135 g per run) |
| **GroupKFold(k=5)** grouped by `segment_key` | Prevents data leakage — no storm spans train and validation |
| **`scale_pos_weight=20`** | Handles 1:21 class imbalance without oversampling |
| **Threshold 0.71** | Tuned on OOF predictions to maximise F1 |
| **CPU-only, no cloud** | Minimal CO₂ footprint — 0.34 g total for full training |

---

## Environmental Impact

Tracked with [CodeCarbon](https://github.com/mlco2/codecarbon) at every training run:

- **Total CO₂:** 0.34 g eq. (full 5-fold CV on 56 K samples)
- **Energy:** 1.04 Wh — equivalent to charging a phone for 12 minutes
- **Model selected partly on efficiency:** LightGBM has the best AUC *and* lower CO₂ than XGBoost

---

## Requirements

- Python 3.10+
- Docker (for `make app-build` / `make app-docker`)
- Training data CSV placed at `data/segment_alerts_all_airports_train.csv`
