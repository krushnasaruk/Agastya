#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Running AGASTYA Trajectory Benchmark Evaluation"
echo "============================================================"

python services/ml/src/evaluation/evaluate.py --all-scenarios
