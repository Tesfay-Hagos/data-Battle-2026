# ============================================================
# DataBattle 2026 — Makefile
# ============================================================
# Usage:
#   make install       Install all dependencies
#   make notebook      Convert .py → .ipynb (all notebooks)
#   make lab           Open JupyterLab with all notebooks
#   make run-eda       Run EDA script directly as Python
#   make sync          Two-way sync .py ↔ .ipynb
#   make push-drive    Convert + push all notebooks to Google Drive
#   make clean         Remove generated figures and .ipynb files
#
# Requires: jupytext  (converts percent-format .py to .ipynb)
# Install:  pip install jupytext
# ============================================================

PYTHON    := ../.venv/bin/python3
PIP       := ../.venv/bin/pip
JUPYTER   := ../.venv/bin/jupyter
NBDIR     := notebooks
FIGDIR    := outputs/figures
DRIVE_NB  := gdrive:databattle2026/notebooks

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

## ── sync: two-way sync .py ↔ .ipynb (keeps both up to date) ────────────────
## Useful during development: edit the .ipynb, sync back to .py
.PHONY: sync
sync:
	@for f in $(NBDIR)/*.ipynb; do \
		echo "  Syncing $$f …"; \
		$(PYTHON) -m jupytext --sync "$$f"; \
	done

## ── push-drive: convert .py → .ipynb locally then upload to Google Drive ────
.PHONY: push-drive
push-drive:
	@echo "Converting .py → .ipynb …"
	@mkdir -p /tmp/db2026_nb
	@for f in $(NBDIR)/*.py; do \
		nb="/tmp/db2026_nb/$$(basename $${f%.py}.ipynb)"; \
		echo "  Converting $$(basename $$f) …"; \
		$(PYTHON) -m jupytext --to notebook --output "$$nb" "$$f"; \
	done
	@echo "Uploading notebooks to Google Drive …"
	@rclone copy /tmp/db2026_nb $(DRIVE_NB) --progress
	@echo "Uploading env_setup.py to Google Drive …"
	@rclone copyto env_setup.py gdrive:databattle2026/env_setup.py
	@echo "Done. Open from Google Drive → databattle2026/notebooks/"

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
	@echo "  make install     Install Python dependencies"
	@echo "  make notebook    Convert all .py notebooks → .ipynb"
	@echo "  make lab         Convert + open JupyterLab"
	@echo "  make run-eda       Run 01_eda.py directly"
	@echo "  make run-final-eda Run 03_final_eda_and_features.py (no plot windows)"
	@echo "  make sync        Two-way sync .py ↔ .ipynb"
	@echo "  make push-drive  Convert + push all notebooks to Google Drive"
	@echo "  make clean       Remove generated .ipynb and figures"
	@echo ""
