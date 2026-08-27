@echo off
echo ============================================================
echo Starting AGASTYA AI Dead Reckoning Telemetry Backend
echo ============================================================

start "AGASTYA API Server" cmd /k "python -m uvicorn services.api.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting Frontend Avionics Mission Control...
cd frontend
call npm run dev
