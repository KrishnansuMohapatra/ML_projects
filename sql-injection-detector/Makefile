.PHONY: help install dev api streamlit test lint train-dev dataset docker

PY ?= python
UV ?= uv

help:                     ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:                  ## Install/sync dependencies (uv)
	$(UV) sync

dev:                      ## Run the API with autoreload on :8000
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api:                      ## Run the API without reload
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000

streamlit:                ## Run the original Streamlit UI
	$(UV) run streamlit run src/app.py

test:                     ## Run the test suite
	$(UV) run pytest -q

lint:                     ## Byte-compile every module as a cheap syntax check
	$(UV) run python -m compileall -q api src scripts

dataset:                  ## Generate the synthetic dev dataset (no Kaggle download)
	$(UV) run $(PY) scripts/make_dev_dataset.py --rows 12000

train-dev: dataset        ## Train a model on the synthetic dev dataset
	$(UV) run $(PY) scripts/train_dev_model.py

docker:                   ## Build the container image
	docker build -t sql-injection-detector-api .
