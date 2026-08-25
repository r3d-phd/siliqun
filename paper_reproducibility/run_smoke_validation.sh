#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
pytest -q tests/test_lindblad_regression.py
python paper_reproducibility/regenerate_solver_validation.py
