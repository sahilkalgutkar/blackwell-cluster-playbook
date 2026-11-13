PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test lint format demo fixtures stack clean

install: $(BIN)/bcp

$(BIN)/bcp: pyproject.toml
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q -e '.[dev]'
	@touch $@

test: install
	$(BIN)/pytest -q

lint: install
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: install
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

# Validate every fixture without GPUs. The degraded node must fail; the others must pass.
demo: install
	PATH="$(abspath $(BIN)):$$PATH" scripts/demo.sh

fixtures: install
	$(BIN)/python scripts/make_fixtures.py

stack: install
	$(BIN)/bcp stack check

clean:
	rm -rf build .pytest_cache .ruff_cache
