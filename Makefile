
.PHONY: help ensure-uv install test test-all lint format clean

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

install: | ensure-uv
	uv sync --all-extras --compile-bytecode
	uv pip install --python .venv/bin/python -e ./third_party/afterburner

test:
	uv run pytest tests/ -v

test-all:
	uv run pytest tests/ -v -m ""

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix
