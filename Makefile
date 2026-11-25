# Common tasks. Run `make help` for the list; docs/experiments.md explains the experiments.
PYTHON ?= python
PAPER_DIR := reports/paper
RUN := $(PYTHON) -m frauddetect run

.PHONY: review-reruns help install install-all lint format test test-all data validate models smoke \
        eda reproduce benchmark tuned learning-curve fairness decision explain monitor \
        small-bank app-data experiments pipeline-check app fonts report clean all

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# ------------------------------------------------------------------ setup
install: ## Core + boosted trees + dev tools
	$(PYTHON) -m pip install -e ".[gbdt,data,report,dev]"

install-all: ## Everything, including deep learning and foundation models (GPU recommended)
	$(PYTHON) -m pip install -e ".[gbdt,dl,fm,explain,fair,data,app,report,dev]"

lint: ## Static checks
	$(PYTHON) -m ruff check src tests app
	$(PYTHON) -m ruff format --check src tests app

format: ## Auto-format
	$(PYTHON) -m ruff check --fix src tests app
	$(PYTHON) -m ruff format src tests app

test: ## Fast unit tests
	$(PYTHON) -m pytest -m "not slow"

test-all: ## All tests, including deep-learning wrappers and the synthetic pipeline
	$(PYTHON) -m pytest

data: ## Download BAF from Kaggle into data/raw and convert to Parquet
	$(PYTHON) -m frauddetect download --remove-csv

validate: ## Validate the downloaded variants
	$(PYTHON) -m frauddetect validate

models: ## List the ten models and whether their dependencies are installed
	$(PYTHON) -m frauddetect models

smoke: ## Check the GPU and fit every installed model on tiny synthetic data
	$(PYTHON) -m frauddetect smoke

# ------------------------------------------------------------------ experiments (real data)
eda: ## E0  EDA of BAF (+ Taiwan statistics if data/external has them)
	$(RUN) eda

reproduce: ## E1  Reproduce BAF Figure 1 (random-search LightGBM, resumable)
	$(RUN) reproduce

benchmark: ## E2  Ten models, default parameters
	$(RUN) benchmark

tuned: ## E3  Equal-budget Optuna tuning (resumable)
	$(RUN) tuned

learning-curve: ## E4  Performance vs training-set size (resumable)
	$(RUN) learning-curve

fairness: ## E5-E6  Age fairness, cutoff sensitivity, mitigation
	$(RUN) fairness

decision: ## E7  Three-tier decision model
	$(RUN) decision

explain: ## E8  SHAP and surrogate rules
	$(RUN) explain

monitor: ## E9  Drift (PSI), monthly tracking, model card
	$(RUN) monitor

small-bank: ## E10 Small-bank dilemma: local vs federated learning
	$(RUN) small-bank

app-data: ## E11 Export aggregated histogram for the demo
	$(RUN) app-data

experiments: ## Run E0-E11 in order (long; use tmux)
	$(RUN) all

review-reruns: ## Re-run the experiments changed after the 11/24 reviews (resumable)
	$(RUN) reproduce
	$(RUN) tuned
	$(RUN) learning-curve
	$(RUN) fairness
	$(RUN) decision
	$(RUN) small-bank

pipeline-check: ## Run every experiment on synthetic data in quick mode (minutes, no real data)
	$(RUN) all --synthetic --quick

app: ## Start the Streamlit demo
	$(PYTHON) -m streamlit run app/streamlit_app.py

# ------------------------------------------------------------------ report
FONT_URL := https://raw.githubusercontent.com/notofonts/noto-cjk/main
FONTS := Serif/OTF/TraditionalChinese/NotoSerifCJKtc-Regular.otf \
         Serif/OTF/TraditionalChinese/NotoSerifCJKtc-Bold.otf \
         Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf \
         Sans/OTF/TraditionalChinese/NotoSansCJKtc-Bold.otf

fonts: ## Download Noto CJK TC fonts (SIL OFL) for the PDF and figures
	mkdir -p $(PAPER_DIR)/fonts
	for f in $(FONTS); do \
	  test -f $(PAPER_DIR)/fonts/$$(basename $$f) || curl -fsSL -o $(PAPER_DIR)/fonts/$$(basename $$f) $(FONT_URL)/$$f; \
	done

report: fonts ## Render the paper (PDF via Typst + Word) with Quarto
	cd $(PAPER_DIR) && quarto render

clean: ## Remove caches and build output (keeps results and data)
	rm -rf .pytest_cache .ruff_cache $(PAPER_DIR)/_output $(PAPER_DIR)/.quarto
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint test validate experiments report ## Everything: checks, experiments, report
