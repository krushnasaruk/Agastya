@echo off
echo ============================================================
echo Running AGASTYA Trajectory Benchmark Evaluation
echo ============================================================

python services\ml\src\evaluation\evaluate.py --all-scenarios
pause
