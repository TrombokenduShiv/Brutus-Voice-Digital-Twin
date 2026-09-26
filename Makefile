.PHONY: install test lint api
install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

api:
	uvicorn voice_twin.api.server:app --host 0.0.0.0 --port 8787
