.PHONY: install install-dev lint format test train download prepare inference upload clean help

PYTHON ?= python
UV := uv

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies (training + dev)
	$(UV) pip install -e ".[train,dev]"

install-dev:  ## Install dev dependencies only
	$(UV) pip install -e ".[dev]"

lint:  ## Run ruff linter
	ruff check scripts/ tests/
	ruff format --check scripts/ tests/

format:  ## Format code with ruff
	ruff format scripts/ tests/
	ruff check --fix scripts/ tests/

test:  ## Run all tests
	pytest tests/ -v

download:  ## Download Common Voice PT-BR dataset (100 hours)
	$(PYTHON) scripts/download_dataset.py --output-dir data/raw --max-hours 100

prepare:  ## Prepare dataset (resample, normalize, split)
	$(PYTHON) scripts/prepare_dataset.py --input-csv data/raw/metadata.csv --output-dir data/processed

train:  ## Fine-tune F5-TTS on PT-BR data
	$(PYTHON) scripts/train.py --dataset-csv data/processed/train.csv --output-dir checkpoints

inference:  ## Run inference test with sample texts
	$(PYTHON) scripts/inference.py --output outputs/

upload:  ## Upload trained model to HuggingFace
	$(PYTHON) scripts/upload_to_hf.py --checkpoint checkpoints/model_last.safetensors

pipeline:  ## Run full pipeline (download → prepare → train → test)
	$(MAKE) download
	$(MAKE) prepare
	$(MAKE) train
	$(MAKE) inference

clean:  ## Remove generated artifacts
	rm -rf data/raw data/processed checkpoints outputs .cache __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
