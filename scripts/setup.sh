#!/usr/bin/env bash
set -e

echo "============================================================"
echo "AGASTYA AI Dead Reckoning - Automated Environment Setup"
echo "============================================================"

echo "[1/4] Checking Python environment..."
python -m pip install --upgrade pip
python -m pip install -r services/navigation-engine/requirements.txt
python -m pip install -r services/ml/requirements.txt
python -m pip install -r services/api/requirements.txt

echo "[2/4] Installing Frontend dependencies..."
cd frontend
npm install
cd ..

echo "[3/4] Running Unit Tests..."
python -m pytest services/navigation-engine/tests/

echo "[4/4] Pre-training default neural dead reckoning checkpoint..."
python services/ml/src/training/train.py --epochs 3 --batch-size 32

echo "============================================================"
echo "Setup Complete! Run ./scripts/run-simulation.sh to launch."
echo "============================================================"
