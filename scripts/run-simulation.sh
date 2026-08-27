#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Starting AGASTYA AI Dead Reckoning Backend & Avionics UI"
echo "============================================================"

# Start backend in background
python -m uvicorn services.api.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd frontend
npm run dev

# Cleanup on exit
kill $BACKEND_PID
