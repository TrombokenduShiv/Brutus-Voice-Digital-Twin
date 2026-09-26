.PHONY: install install-ml install-vocoder test test-ml validate-ml smoke-train lint api

install:
	python -m pip install -e ".[dev]"

install-ml:
	python -m pip install -e ".[dev,ml]"

install-vocoder:
	python -m pip install -e ".[vocoder]"

test:
	pytest

test-ml:
	python -m compileall -q voice_twin training evaluation tests scripts
	pytest

validate-ml:
	python scripts/validate_training_setup.py

smoke-train:
	python scripts/smoke_train.py

lint:
	ruff check .

api:
	uvicorn voice_twin.api.server:app --host 0.0.0.0 --port 8787
