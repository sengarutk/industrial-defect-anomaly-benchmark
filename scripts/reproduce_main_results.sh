#!/usr/bin/env bash
# End-to-end reproduction: data -> tests -> benchmark -> analyses -> figures -> report -> paper assets.
# Usage: PYTHON=/path/to/python scripts/reproduce_main_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== 1. Environment"
"$PY" scripts/collect_environment.py

echo "== 2. Data (download if needed, verify against the official manifest)"
"$PY" scripts/download_dataset.py

echo "== 3. Tests"
"$PY" -m pytest -q tests

echo "== 4. Benchmark sweep (resumes; pass --force to rerun everything)"
"$PY" scripts/run_benchmark.py

echo "== 5. Operating-point evaluation, CCT, aggregation ablation, failure cases"
"$PY" scripts/run_operational_eval.py
"$PY" scripts/run_cct_experiments.py
"$PY" scripts/run_ablations.py
"$PY" scripts/run_failure_analysis.py

echo "== 6. GPU-only measurements (skipped when CUDA is unavailable)"
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  "$PY" scripts/profile_latency.py
  "$PY" scripts/run_coreset_benchmark.py
else
  echo "   CUDA not available: latency and coreset benchmark not measured"
fi

echo "== 7. Figures, report, paper assets"
"$PY" scripts/generate_figures.py
"$PY" scripts/generate_report.py
"$PY" scripts/sync_paper_assets.py --zip
echo "Done."
