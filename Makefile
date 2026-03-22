# ============================================================
# DataBattle 2026 — Makefile
# ============================================================
# Usage:
#   make install       Install all dependencies
#   make notebook      Convert .py → .ipynb (all notebooks)
#   make lab           Open JupyterLab with all notebooks
#   make run-eda       Run EDA script directly as Python
#   make sync          Two-way sync .py ↔ .ipynb
#   make clean         Remove generated figures and .ipynb files
#
# Requires: jupytext  (converts percent-format .py to .ipynb)
# Install:  pip install jupytext
# ============================================================

PYTHON    := .venv/bin/python3
PIP       := .venv/bin/pip
NBDIR     := notebooks
FIGDIR    := outputs/figures

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
		jupytext --to notebook --output "$${f%.py}.ipynb" "$$f"; \
	done
	@echo "Done. Open notebooks/ in JupyterLab."

## ── single target: convert a specific notebook ──────────────────────────────
$(NBDIR)/%.ipynb: $(NBDIR)/%.py
	jupytext --to notebook --output $@ $<

## ── lab: convert .py → .ipynb then open JupyterLab ─────────────────────────
.PHONY: lab
lab: notebook
	@echo "Opening JupyterLab …"
	.venv/bin/jupyter lab $(NBDIR)/

## ── run-eda: execute EDA script directly (no Jupyter needed) ────────────────
.PHONY: run-eda
run-eda:
	@echo "Running EDA …"
	$(PYTHON) $(NBDIR)/01_eda.py
	@echo "Figures saved to $(FIGDIR)/"

## ── sync: two-way sync .py ↔ .ipynb (keeps both up to date) ────────────────
## Useful during development: edit the .ipynb, sync back to .py
.PHONY: sync
sync:
	@for f in $(NBDIR)/*.ipynb; do \
		echo "  Syncing $$f …"; \
		jupytext --sync "$$f"; \
	done

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
	@echo "  make run-eda     Run EDA script directly"
	@echo "  make sync        Two-way sync .py ↔ .ipynb"
	@echo "  make clean       Remove generated .ipynb and figures"
	@echo ""
