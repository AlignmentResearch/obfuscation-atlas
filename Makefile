PYTHON ?= python3
USE_SYSTEM ?=

ifeq ($(USE_SYSTEM),1)
    ENV := $(shell $(PYTHON) -c "import sys; print(sys.prefix)")
    VENV_DEP :=
    RUNNER :=
else
    ENV := $(PWD)/.venv
    VENV_DEP := .venv/bin/python .venv/bin/activate
    RUNNER := uv run
endif

.PHONY: help install test test-all lint format clean ensure-uv

help:
	@echo "Available commands:"
	@echo "  make install          Install all dependencies"
	@echo "  make test             Run fast tests"
	@echo "  make test-all         Run all tests including slow/integration"
	@echo "  make lint             Run linters"
	@echo "  make format           Auto-format code"
	@echo "  make clean            Remove build artifacts and caches"

ensure-uv:
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)

install: $(VENV_DEP) | ensure-uv
ifeq ($(USE_SYSTEM),1)
	UV_PROJECT_ENVIRONMENT=$(ENV) uv pip install -e ./third_party/afterburner
	UV_PROJECT_ENVIRONMENT=$(ENV) uv sync --all-extras --compile-bytecode --inexact
endif

.venv/bin/python .venv/bin/activate: pyproject.toml | ensure-uv
	UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=$(PWD)/.venv uv pip install -e ./third_party/afterburner
	UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=$(PWD)/.venv uv sync --all-extras --compile-bytecode --inexact
	@touch .venv/bin/python .venv/bin/activate

test: $(VENV_DEP)
	$(RUNNER) pytest tests/ -v

test-all: $(VENV_DEP)
	$(RUNNER) pytest tests/ -v -m ""

lint: $(VENV_DEP)
	$(RUNNER) ruff check .

format: $(VENV_DEP)
	$(RUNNER) ruff check . --fix

clean:
	@echo "Cleaning build artifacts and caches..."
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache build dist 2>/dev/null || true
	@echo "Clean complete!"
