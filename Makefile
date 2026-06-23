.PHONY: check smoke test compile env clean-cache

env:
	python scripts/00_check_environment.py

smoke:
	python scripts/run_smoke_test.py

test:
	python -m pytest -q

compile:
	python -m compileall src scripts

check: env smoke test compile

clean-cache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
