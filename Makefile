# Spam Detector — common tasks.
# `make help` lists every target. Works on Linux/macOS/Git Bash and via
# `make` on Windows (winget/scoop install make).

PYTHON ?= python
PORT_API ?= 8000
PORT_UI ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install data train retrain test coverage lint format fix check \
        ui api docker docker-down clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in editable mode with every extra
	$(PYTHON) -m pip install --editable ".[all]"

data: ## Download the UCI SMS Spam Collection into data/
	$(PYTHON) -m spam_detector.data

train: ## Train the model and write models/spam_model.joblib + metrics.json
	$(PYTHON) -m spam_detector.train

retrain: data train ## Re-download the dataset and retrain from scratch

test: ## Run the test suite
	$(PYTHON) -m pytest -q

coverage: ## Run tests with coverage (terminal + htmlcov/)
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=html

lint: ## Check lint and formatting (no writes)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply lint fixes and formatting
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

fix: format ## Alias for `format`

check: lint test ## What CI runs

ui: ## Start the Streamlit demo
	$(PYTHON) -m streamlit run src/spam_detector/app.py --server.port $(PORT_UI)

api: ## Start the FastAPI service with autoreload
	$(PYTHON) -m uvicorn spam_detector.api:app --reload --port $(PORT_API)

docker: ## Build and start API + UI containers
	docker compose up --build

docker-down: ## Stop the containers
	docker compose down

clean: ## Remove caches and generated reports
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
