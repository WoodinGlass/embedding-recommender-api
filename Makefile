# =============================================================================
# embedding-recommender-api — developer commands
#
# Recipes use tabs. Edit with an editor that preserves them (or use
# `make -f <(sed 's/^    /\t/' <file>)` in a pinch).
# =============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

PY       ?= python
PIP      ?= $(PY) -m pip
SRC      := src
TESTS    := tests
SCRIPTS  := scripts

# -----------------------------------------------------------------------------
# meta
# -----------------------------------------------------------------------------
.PHONY: help
help:  ## Show available targets.
	awk 'BEGIN {FS = ":.*?##"; printf "\nUsage: make <target>\n\nTargets:\n"} \
	     /^[a-zA-Z_0-9-]+:.*?##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# -----------------------------------------------------------------------------
# install
# -----------------------------------------------------------------------------
.PHONY: install-dev
install-dev:  ## Install the package with the dev extra (editable).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: install-hooks
install-hooks:  ## Install pre-commit git hooks.
	pre-commit install --install-hooks

# -----------------------------------------------------------------------------
# quality
# -----------------------------------------------------------------------------
.PHONY: fmt
fmt:  ## Auto-format and auto-fix Python.
	$(PY) -m ruff check --fix $(SRC) $(TESTS) $(SCRIPTS)
	$(PY) -m ruff format $(SRC) $(TESTS) $(SCRIPTS)

.PHONY: lint
lint:  ## Lint without modifying files.
	$(PY) -m ruff check $(SRC) $(TESTS) $(SCRIPTS)
	$(PY) -m ruff format --check $(SRC) $(TESTS) $(SCRIPTS)

.PHONY: typecheck
typecheck:  ## Static type checking (mypy, strict).
	$(PY) -m mypy $(SRC) $(TESTS) $(SCRIPTS)

.PHONY: check-markers
check-markers:  ## Enforce test-tier marker discipline.
	$(PY) $(SCRIPTS)/check_markers.py

.PHONY: check-ruff-version
check-ruff-version:  ## Enforce ruff version == pyproject pin (ADR-0025).
	$(PY) $(SCRIPTS)/check_ruff_version.py

.PHONY: check
check: lint typecheck check-markers check-ruff-version test-unit  ## Run all fast quality gates.

# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------
.PHONY: test
test:  ## All tests except load; integration skips if env not set.
	$(PY) -m pytest -q -m "not load"

.PHONY: test-unit
test-unit:  ## Unit tests only. `--ff` runs the last failure first.
	$(PY) -m pytest -q -m unit --ff

.PHONY: test-integration
test-integration:  ## Integration tests, light tier (postgres/redis). Skips encoder parity.
	$(PY) -m pytest -q -m "integration and not encoder"

.PHONY: test-encoder
test-encoder:  ## Encoder parity tests. Requires the [inference,export] extra.
	$(PY) -m pytest -q -m "integration and encoder"

.PHONY: eval
eval:  ## Run the offline evaluation and enforce the threshold gate (ADR-0010).
	$(PY) scripts/eval.py

.PHONY: migrate
migrate:  ## Apply Alembic migrations (reads DATABASE_URL from the environment).
	$(PY) -m alembic upgrade head

.PHONY: popularity-refresh
popularity-refresh:  ## Refresh popularity_snapshot from the item table (ADR-0020).
	$(PY) $(SCRIPTS)/refresh_popularity.py

.PHONY: index-build
index-build:  ## Build an index from the active embedding run.
	$(PY) scripts/build_index.py

.PHONY: index-promote
index-promote:  ## Promote an index. Usage: make index-promote VERSION=idx-xxxx
	@test -n "$(VERSION)" || (echo "usage: make index-promote VERSION=idx-xxxx" && exit 2)
	$(PY) scripts/promote_index.py --index-version "$(VERSION)"

.PHONY: index-rollback
index-rollback:  ## Revert to the most recently retired index.
	$(PY) scripts/rollback_index.py

.PHONY: bench-faiss
bench-faiss:  ## Run the FAISS benchmark (requires the [bench] extra) and render it.
	$(PY) scripts/bench_faiss.py
	$(PY) scripts/render_benchmark.py

.PHONY: coverage
coverage:  ## Unit tests with coverage report.
	$(PY) -m pytest -q -m unit --cov=src/recsys --cov-report=term-missing

# -----------------------------------------------------------------------------
# local stack (Docker; see README for caveats on Colab)
# -----------------------------------------------------------------------------
.PHONY: up
up:  ## Start the local stack (Docker Compose).
	docker compose up -d --build

.PHONY: down
down:  ## Stop the local stack and remove volumes.
	docker compose down -v

# -----------------------------------------------------------------------------
# housekeeping
# -----------------------------------------------------------------------------
.PHONY: clean
clean:  ## Remove caches and build artifacts.
	rm -rf build dist ./*.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
