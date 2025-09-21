format:
	black src/ && isort src/

lint:
	flake8 src/ && black --check src/ && isort --check-only src/

test:
	pytest tests/