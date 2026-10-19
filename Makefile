.PHONY: sync lint test check

sync:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

check: lint test
