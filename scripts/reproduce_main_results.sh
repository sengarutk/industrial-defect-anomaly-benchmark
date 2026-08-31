#!/usr/bin/env bash
set -e
echo "=== 1. Verifying Hardware & Environment Metadata ==="
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "=== 2. Running Comprehensive Pytest Suite with Coverage ==="
pytest --cov=src --cov-report=term-missing tests/ -v

echo "=== 3. Executing Operational Stream Evaluations (IID, Burst, Drift) ==="
python scripts/run_operational_eval.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad

echo "=== 4. Running Image Aggregation & Coreset Systems Ablations ==="
python scripts/run_ablations.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad

echo "=== 5. Compiling Publication LaTeX Tables and Figures ==="
python scripts/generate_plots.py --tables-dir results/mvtec_ad/tables --output-dir results/mvtec_ad/figures
python scripts/generate_operational_plots.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad/figures/operational
python scripts/generate_report.py --tables-dir results/mvtec_ad/tables --docs-dir docs

echo "=== ✅ Master Benchmark Reproduction Successfully Completed ==="