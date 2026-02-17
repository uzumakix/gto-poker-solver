.PHONY: run test lint clean

run:
	python main.py

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check gto_poker_solver/ tests/ main.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
