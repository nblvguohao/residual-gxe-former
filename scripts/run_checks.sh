#!/usr/bin/env bash
set -euo pipefail
python scripts/00_check_environment.py
python scripts/run_smoke_test.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall src scripts
