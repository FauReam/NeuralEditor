.PHONY: install test lint format clean run download-model train-lora web

install:
	pip install -e ".[dev,train]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

download-model:
	python scripts/download_model.py

train-lora:
	python scripts/train_lora.py \
		--data data/romance_chat_sample.jsonl \
		--output lora_romance \
		--epochs 1

run:
	python -m src.main

run-dev:
	python -m src.main --character config/characters/default.yaml

web:
	python -m src.web.server
