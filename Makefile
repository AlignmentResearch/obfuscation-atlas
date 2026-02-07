
.PHONY: help install test test-all lint format clean

help:
	@echo "Available commands:"
	@echo "  make install          Install all dependencies"
	@echo "  make test             Run fast tests"
	@echo "  make test-all         Run all tests including slow/integration"
	@echo "  make lint             Run linters"
	@echo "  make format           Auto-format code"
	@echo "  make clean            Remove build artifacts and caches"

install:
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
