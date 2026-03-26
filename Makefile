# ============================================================
# DataBattle 2026 — Makefile
# ============================================================
# Usage:
#   make install         Install all dependencies
#   make notebook        Convert .py → .ipynb (all notebooks)
#   make lab             Open JupyterLab with all notebooks
#   make run-eda         Run EDA script directly as Python
#   make run-final-eda   Run 03_final_eda_and_features.py
#   make sync            Two-way sync .py ↔ .ipynb
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

## ── install: install all Python dependencies ────────────────────────────────
.PHONY: install
install:
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

## ── run-eda: execute EDA script directly (no Jupyter needed) ────────────────
.PHONY: run-eda
run-eda:
	@echo "Running EDA …"
	$(PYTHON) $(NBDIR)/01_eda.py
	@echo "Figures saved to $(FIGDIR)/"

## ── run-final-eda: run 03_final_eda_and_features.py without opening any plots
.PHONY: run-final-eda
run-final-eda:
	@echo "Running final EDA (no plot windows) …"
	MPLBACKEND=Agg $(PYTHON) $(NBDIR)/03_final_eda_and_features.py
	@echo "Done. Figures saved to outputs/figures/"

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
	rm -f $(FIGDIR)/*.png $(FIGDIR)/*.html
	@echo "Cleaned generated files."

## ── help ────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "DataBattle 2026 — available make targets:"
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
	@echo "    make run-eda         Run 01_eda.py directly"
	@echo "    make run-final-eda   Run 03_final_eda_and_features.py"
	@echo "    make sync            Two-way sync .py ↔ .ipynb"
	@echo ""
	@echo "  Drive:"
	@echo "    make push-drive      Convert notebooks + push notebooks/ and src/ to Drive"
	@echo "    make pull-drive      Pull notebooks + figures from Google Drive"
	@echo ""
	@echo "  Other:"
	@echo "    make install         Install Python dependencies"
	@echo "    make clean           Remove generated .ipynb and figures"
	@echo ""
