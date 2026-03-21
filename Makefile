.PHONY: test test-unit test-agent lint

# Unit tests only (no LLM calls, fast) with coverage
test-unit:
	python -m pytest tests/ --ignore=tests/agent -v \
		--cov --cov-report=term-missing --cov-report=html

# Agent integration tests (requires LLM API)
test-agent:
	python -m pytest tests/agent -v

lint:
	ruff check .
