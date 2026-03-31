# ============================================================
# DataBattle 2026 — Makefile
# ============================================================
# Usage:
#   make install         Install all dependencies
#   make notebook        Convert .py → .ipynb (all notebooks)
#   make lab             Open JupyterLab with all notebooks
#   make run FILE=path   Run any .py file (src/ or notebooks/)
#   make run-eda         Run EDA script directly as Python
#   make run-final-eda   Run 03_final_eda_and_features.py
#   make run-compare     Run notebook 06 model comparison locally
#   make run-shap        Run notebook 05 SHAP explainability locally
#   make sync            Two-way sync .py ↔ .ipynb
#   make tune            Optuna hyperparameter search (saves best_params.json)
#   make pipeline        Run full pipeline: features→train→evaluate→predict
#   make train           Build features + train LightGBM (CV + temporal test)
#   make evaluate        Print OOF + gain/risk report from saved results
#   make predict         Generate predictions.csv for test data submission
#   make push-drive      Convert + push notebooks AND src/ to Google Drive
#   make pull-drive      Pull notebooks + figures from Google Drive
#   make clean           Remove generated figures and .ipynb files
#
# Requires: jupytext  (converts percent-format .py to .ipynb)
# Install:  pip install jupytext
# ============================================================

PYTHON    := ../.venv/bin/python3
PIP       := ../.venv/bin/pip
JUPYTER   := ../.venv/bin/jupyter
NBDIR     := notebooks
SRCDIR    := src
FIGDIR    := outputs/figures
DRIVE_NB   := gdrive:databattle2026/notebooks
DRIVE_SRC  := gdrive:databattle2026/src
TEST_DATA := /home/tesfayh/Documents/projects/personal/data-computition/dataset_test/dataset_set.csv
SUBMISSION := outputs/submissions/predictions.csv

## ── install: create venv if needed, then install all Python dependencies ─────
.PHONY: install
install:
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "Creating virtual environment at ../.venv …"; \
		python3 -m venv ../.venv; \
	fi
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt
	$(PIP) install jupytext --upgrade

## ── notebook: convert all .py notebooks → .ipynb ───────────────────────────
.PHONY: notebook
notebook:
	@echo "Converting .py → .ipynb …"
	@for f in $(NBDIR)/*.py; do \
		echo "  Converting $$f …"; \
		$(PYTHON) -m jupytext --to notebook --output "$${f%.py}.ipynb" "$$f"; \
	done
	@echo "Done. Open notebooks/ in JupyterLab."

## ── single target: convert a specific notebook ──────────────────────────────
$(NBDIR)/%.ipynb: $(NBDIR)/%.py
	$(PYTHON) -m jupytext --to notebook --output $@ $<

## ── lab: convert .py → .ipynb then open JupyterLab ─────────────────────────
.PHONY: lab
lab: notebook
	@$(PYTHON) -m ipykernel install --user --name=databattle2026 --display-name="DataBattle 2026" --quiet 2>/dev/null || true
	@echo "Opening JupyterLab …"
	$(JUPYTER) lab --notebook-dir=.

## ── check-all: syntax-check every src/, notebooks/, and app/ .py file ──────
.PHONY: check-all
check-all:
	@echo "════════════════════════════════════════════════"
	@echo " Syntax check — src/ + notebooks/ + app/"
	@echo "════════════════════════════════════════════════"
	@pass=0; fail=0; \
	for f in $(NBDIR)/0*.py $(SRCDIR)/*.py app/Home.py app/utils/loaders.py app/pages/*.py; do \
		result=$$($(PYTHON) -m py_compile "$$f" 2>&1); \
		if [ $$? -eq 0 ]; then \
			echo "  OK   $$f"; pass=$$((pass+1)); \
		else \
			echo "  FAIL $$f"; \
			echo "       $$result"; fail=$$((fail+1)); \
		fi; \
	done; \
	echo ""; \
	echo "  $$pass passed, $$fail failed"; \
	[ $$fail -eq 0 ]

## ── run: execute any .py file — usage: make run FILE=src/compare_models.py ──
## Works for both src/ scripts and notebooks/ .py files.
.PHONY: run
run:
ifndef FILE
	@echo "Usage: make run FILE=<path/to/script.py>"
	@echo "  e.g. make run FILE=src/compare_models.py"
	@echo "  e.g. make run FILE=notebooks/06_model_comparison.py"
	@exit 1
endif
	@echo "════════════════════════════════════════════════"
	@echo " Running: $(FILE)"
	@echo "════════════════════════════════════════════════"
	MPLBACKEND=Agg $(PYTHON) $(FILE)
	@echo "Done."

## ── run-compare: run notebook 06 model comparison locally ───────────────────
.PHONY: run-compare
run-compare:
	@echo "════════════════════════════════════════════════"
	@echo " Model Comparison (notebook 06) — running locally"
	@echo " Figures → outputs/figures/model-comparison/"
	@echo "════════════════════════════════════════════════"
	MPLBACKEND=Agg $(PYTHON) $(NBDIR)/06_model_comparison.py

## ── run-shap: run notebook 05 SHAP explainability locally ───────────────────
.PHONY: run-shap
run-shap:
	@echo "════════════════════════════════════════════════"
	@echo " SHAP Explainability (notebook 05) — running locally"
	@echo " Figures → outputs/figures/shap/"
	@echo "════════════════════════════════════════════════"
	MPLBACKEND=Agg $(PYTHON) $(NBDIR)/05_shap_explainability.py

## ── run-eda: execute EDA script directly (no Jupyter needed) ────────────────
.PHONY: run-eda
run-eda:
	@echo "Running EDA …"
	$(PYTHON) $(NBDIR)/01_eda.py
	@echo "Figures saved to $(FIGDIR)/eda/"

## ── run-final-eda: run 03_final_eda_and_features.py without opening any plots
.PHONY: run-final-eda
run-final-eda:
	@echo "Running final EDA (no plot windows) …"
	MPLBACKEND=Agg $(PYTHON) $(NBDIR)/03_final_eda_and_features.py
	@echo "Done. Figures saved to outputs/figures/"

## ── compare: run model comparison (LR vs XGBoost vs LightGBM) ───────────────
.PHONY: compare
compare:
	@echo "══════════════════════════════════════════════════"
	@echo " Model Comparison: LR vs XGBoost vs LightGBM"
	@echo "══════════════════════════════════════════════════"
	MPLBACKEND=Agg $(PYTHON) $(SRCDIR)/compare_models.py

## ── tune: Optuna hyperparameter search ──────────────────────────────────────
.PHONY: tune
tune:
	@echo "══════════════════════════════════════════════════"
	@echo " Optuna Hyperparameter Search (50 trials)"
	@echo "══════════════════════════════════════════════════"
	$(PYTHON) $(SRCDIR)/tune.py

## ── pipeline: run full pipeline end-to-end ──────────────────────────────────
## features → train (CV + temporal) → evaluate (OOF + gain/risk) → predict
.PHONY: pipeline
pipeline: train evaluate predict

## ── train: build features + train LightGBM ──────────────────────────────────
.PHONY: train
train:
	@echo "══════════════════════════════════════════════════"
	@echo " Step 1/3 — Feature engineering + Training"
	@echo "══════════════════════════════════════════════════"
	MPLBACKEND=Agg $(PYTHON) $(SRCDIR)/train.py

## ── evaluate: print OOF and gain/risk report ────────────────────────────────
.PHONY: evaluate
evaluate:
	@echo "══════════════════════════════════════════════════"
	@echo " Step 2/3 — Evaluation report"
	@echo "══════════════════════════════════════════════════"
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); \
	              from evaluate import full_report; full_report()"

## ── predict: generate submission predictions on test data ───────────────────
.PHONY: predict
predict:
	@echo "══════════════════════════════════════════════════"
	@echo " Step 3/3 — Generating predictions.csv"
	@echo "══════════════════════════════════════════════════"
	$(PYTHON) $(SRCDIR)/predict.py \
		--test $(TEST_DATA) \
		--output $(SUBMISSION)

## ── sync: two-way sync .py ↔ .ipynb (keeps both up to date) ────────────────
## Useful during development: edit the .ipynb, sync back to .py
.PHONY: sync
sync:
	@for f in $(NBDIR)/*.ipynb; do \
		echo "  Syncing $$f …"; \
		$(PYTHON) -m jupytext --sync "$$f"; \
	done

## ── push-drive: convert .py → .ipynb + upload notebooks AND src/ to Drive ───
.PHONY: push-drive
push-drive: notebook
	@echo "Uploading notebooks to Google Drive …"
	@rclone copy $(NBDIR) $(DRIVE_NB) --include "*.ipynb" --progress
	@rclone copy $(NBDIR) $(DRIVE_NB) --include "*.py" --progress
	@echo "Uploading src/ to Google Drive …"
	@rclone copy $(SRCDIR) $(DRIVE_SRC) --include "*.py" --progress
	@echo "Uploading env_setup.py to Google Drive …"
	@rclone copyto env_setup.py gdrive:databattle2026/env_setup.py
	@echo "Done. Open from Google Drive → databattle2026/"
	@echo "(data/ not synced — already in Drive, upload manually if changed)"

## ── pull-drive: pull notebooks (with outputs) + figures from Google Drive ────
.PHONY: pull-drive
pull-drive:
	@echo "Pulling notebooks from Google Drive …"
	@rclone copy $(DRIVE_NB) $(NBDIR) --include "*.ipynb" --progress
	@echo "Pulling figures from Google Drive …"
	@rclone copy gdrive:databattle2026/outputs/figures $(FIGDIR) --progress
	@echo "Done. Outputs synced to outputs/figures/"

## ── clean: remove generated outputs ─────────────────────────────────────────
.PHONY: clean
clean:
	rm -f $(NBDIR)/*.ipynb
	rm -f $(FIGDIR)/eda/*.png $(FIGDIR)/eda/*.html
	rm -f $(FIGDIR)/model-comparison/*.png
	rm -f $(FIGDIR)/shap/*.png
	@echo "Cleaned generated files."

## ── app: install deps if needed, then run Streamlit dashboard locally ────────
## Clone the repo and run `make app` — that's all.
.PHONY: app
app:
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "Virtual environment not found — running make install first …"; \
		$(MAKE) install; \
	fi
	@if ! $(PYTHON) -c "import streamlit" 2>/dev/null; then \
		echo "Dependencies not installed — running make install first …"; \
		$(MAKE) install; \
	fi
	@echo ""
	@echo "  ⚡ DataBattle 2026 — starting dashboard"
	@echo "  Local:    http://localhost:8501"
	@echo ""
	DATABATTLE_ROOT=$(shell pwd) MPLBACKEND=Agg \
	$(PYTHON) -m streamlit run app/Home.py \
		--server.port=8501 \
		--browser.gatherUsageStats=false

## ── app-build: build Docker image with all outputs baked in ─────────────────
.PHONY: app-build
app-build:
	@echo "Building Docker image (includes outputs/ and data/) …"
	docker build -t databattle2026-app .

## ── app-docker: run the pre-built Docker image ──────────────────────────────
## Uses --network host (Linux) so the external IP is accessible directly.
## If on Mac/Windows replace --network host with -p 8501:8501
.PHONY: app-docker
app-docker:
	@echo "Starting Docker container …"
	@echo "Local:    http://localhost:8501"
	@echo "External: http://$$(hostname -I | awk '{print $$1}'):8501"
	docker run --rm --network host \
		-e DATABATTLE_ROOT=/app \
		-e MPLBACKEND=Agg \
		databattle2026-app

## ── app-stop: stop running Docker container ──────────────────────────────────
.PHONY: app-stop
app-stop:
	docker ps -q --filter ancestor=databattle2026-app | xargs -r docker stop

## ── help ────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "DataBattle 2026 — available make targets:"
	@echo ""
	@echo "  Run any file:"
	@echo "    make run FILE=src/compare_models.py    run any src/ or notebooks/ .py"
	@echo ""
	@echo "  Model comparison:"
	@echo "    make compare         LR vs XGBoost vs LightGBM via src/compare_models.py"
	@echo "    make run-compare     Same via notebooks/06_model_comparison.py"
	@echo ""
	@echo "  Hyperparameter search (run before pipeline):"
	@echo "    make tune            Optuna search → outputs/saves/best_params.json"
	@echo ""
	@echo "  Pipeline (run in order):"
	@echo "    make pipeline        Full run: train → evaluate → predict"
	@echo "    make train           Feature engineering + LightGBM CV training"
	@echo "    make evaluate        OOF report + gain/risk sweep"
	@echo "    make predict         Generate predictions.csv for test submission"
	@echo ""
	@echo "  Notebooks:"
	@echo "    make notebook        Convert all .py notebooks → .ipynb"
	@echo "    make lab             Convert + open JupyterLab"
	@echo "    make run FILE=notebooks/03_final_eda_and_features.py"
	@echo "    make run-compare     Run notebook 06 locally"
	@echo "    make run-shap        Run notebook 05 locally"
	@echo "    make run-eda         Run 01_eda.py directly"
	@echo "    make run-final-eda   Run 03_final_eda_and_features.py"
	@echo "    make sync            Two-way sync .py ↔ .ipynb"
	@echo ""
	@echo "  Drive:"
	@echo "    make push-drive      Convert notebooks + push notebooks/ and src/ to Drive"
	@echo "    make pull-drive      Pull notebooks + figures from Google Drive"
	@echo ""
	@echo "  Dashboard:"
	@echo "    make app             Install deps if needed + run Streamlit locally"
	@echo "    make app-build       Build Docker image (outputs/ baked in)"
	@echo "    make app-docker      Run Docker image at localhost:8501"
	@echo "    make app-stop        Stop running Docker container"
	@echo ""
	@echo "  Other:"
	@echo "    make check-all       Syntax-check src/ + notebooks/ + app/ .py files"
	@echo "    make install         Install Python dependencies"
	@echo "    make clean           Remove generated .ipynb and figures"
	@echo ""
