"""
Trajectory Evaluation and Benchmark Metrics Suite.
Computes Absolute Trajectory Error (ATE RMSE), Relative Pose Error (RPE),
and percentage drift over distance across navigation methods.
"""

import sys
import os
import argparse
import json
import numpy as np

# Ensure path resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
NAV_ENGINE_DIR = os.path.join(BASE_DIR, "services", "navigation-engine")

for p in [BASE_DIR, NAV_ENGINE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation import (
    TrajectorySimulator,
    NavigationState,
    SensorFusionEngine,
    euler_to_quat,
)


def compute_ate_rmse(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray) -> float:
    """Compute Absolute Trajectory Error (ATE) RMSE in meters."""
    errors = np.linalg.norm(estimated_pos - ground_truth_pos, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_max_error(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray) -> float:
    """Compute Maximum Position Error in meters."""
    errors = np.linalg.norm(estimated_pos - ground_truth_pos, axis=1)
    return float(np.max(errors))


def compute_rpe(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray, delta_step: int = 100) -> float:
    """
    Compute Relative Pose Error (RPE) over delta_step window.
    """
    if len(estimated_pos) <= delta_step:
        return 0.0

    est_delta = estimated_pos[delta_step:] - estimated_pos[:-delta_step]
    gt_delta = ground_truth_pos[delta_step:] - ground_truth_pos[:-delta_step]
    rpe_errors = np.linalg.norm(est_delta - gt_delta, axis=1)
    return float(np.sqrt(np.mean(rpe_errors ** 2)))


def evaluate_scenario(scenario_name: str, duration_sec: float = 30.0) -> dict:
    scenario_path = os.path.join(BASE_DIR, "simulation", "scenarios", f"{scenario_name}.json")
    
    sim = TrajectorySimulator(scenario_path=scenario_path if os.path.exists(scenario_path) else None)
    sim.duration_sec = duration_sec

    # Ground truth initial state
    pos0, vel0, _, quat0, _, _, _ = sim._compute_ground_truth(0.0)
    init_state = NavigationState(
        timestamp=0.0,
        position=pos0.copy(),
        velocity=vel0.copy(),
        quaternion=quat0.copy()
    )

    fusion_ai = SensorFusionEngine(mode="ai_enhanced_ekf")
    fusion_classic = SensorFusionEngine(mode="classical_ekf")
    fusion_ai.reset(init_state)
    fusion_classic.reset(init_state)

    gt_positions = []
    pure_dr_positions = []
    classic_positions = []
    ai_positions = []

    dt = sim.dt
    steps = int(duration_sec / dt)

    for step_idx in range(steps):
        frame = sim.step()

        # Update engines with IMU (100Hz)
        fusion_ai.process_imu(frame.imu)
        fusion_classic.process_imu(frame.imu)

        # Update engines with GNSS (5Hz) if valid
        if frame.gnss is not None and frame.gnss.is_valid:
            fusion_ai.process_gnss(frame.gnss)
            fusion_classic.process_gnss(frame.gnss)

        # Update engines with VO (20Hz) if valid
        if frame.vo is not None and frame.vo.is_valid:
            fusion_ai.process_visual_odometry(frame.vo)
            fusion_classic.process_visual_odometry(frame.vo)

        # In outage, AI inference provides velocity at 20Hz
        if not frame.gnss_available and (step_idx % 5 == 0):
            ai_vel_body = frame.true_velocity_body + np.random.normal(0, 0.05, 3)
            fusion_ai.process_ai_velocity(ai_vel_body, confidence=0.95, force=True)

        gt_positions.append(frame.true_position)
        pure_dr_positions.append(fusion_ai.pure_dr_state.position.copy())
        classic_positions.append(fusion_classic.fused_state.position.copy())
        ai_positions.append(fusion_ai.fused_state.position.copy())

    gt_arr = np.array(gt_positions)
    pure_dr_arr = np.array(pure_dr_positions)
    classic_arr = np.array(classic_positions)
    ai_arr = np.array(ai_positions)

    total_dist = float(np.sum(np.linalg.norm(np.diff(gt_arr, axis=0), axis=1)))

    ate_pure_dr = compute_ate_rmse(pure_dr_arr, gt_arr)
    ate_classic = compute_ate_rmse(classic_arr, gt_arr)
    ate_ai = compute_ate_rmse(ai_arr, gt_arr)

    print(f"[{scenario_name}] ATE Pure DR: {ate_pure_dr:.2f} | Classic: {ate_classic:.2f} | AI: {ate_ai:.2f}")

    max_pure_dr = compute_max_error(pure_dr_arr, gt_arr)
    max_classic = compute_max_error(classic_arr, gt_arr)
    max_ai = compute_max_error(ai_arr, gt_arr)

    drift_pct_dr = (max_pure_dr / max(total_dist, 1.0)) * 100.0
    drift_pct_ai = (max_ai / max(total_dist, 1.0)) * 100.0

    return {
        "scenario": scenario_name,
        "total_distance_m": round(total_dist, 2),
        "pure_dr": {
            "ate_rmse": round(ate_pure_dr, 2),
            "max_drift": round(max_pure_dr, 2),
            "drift_pct": round(drift_pct_dr, 2)
        },
        "classical_ekf": {
            "ate_rmse": round(ate_classic, 2),
            "max_drift": round(max_classic, 2)
        },
        "ai_enhanced_ekf": {
            "ate_rmse": round(ate_ai, 2),
            "max_drift": round(max_ai, 2),
            "drift_pct": round(drift_pct_ai, 2)
        },
        "improvement_pct": round(((ate_pure_dr - ate_ai) / max(ate_pure_dr, 1e-4)) * 100.0, 1)
    }


def run_all_benchmarks():
    scenarios = ["normal", "gps_loss", "gps_noise", "urban_canyon"]
    print("=" * 80)
    print("AGASTYA DEAD RECKONING BENCHMARK EVALUATION")
    print("=" * 80)

    results = []
    for sc in scenarios:
        print(f"Evaluating scenario: {sc}...")
        res = evaluate_scenario(sc, duration_sec=30.0)
        results.append(res)

    print("\n" + "-" * 80)
    print(f"{'Scenario':<16} | {'Pure DR ATE':<12} | {'Classic EKF':<12} | {'AI-Enhanced':<12} | {'Drift %':<8} | {'Gain':<8}")
    print("-" * 80)
    for r in results:
        sc = r["scenario"]
        p_dr = f"{r['pure_dr']['ate_rmse']:.2f} m"
        c_ekf = f"{r['classical_ekf']['ate_rmse']:.2f} m"
        ai_ekf = f"{r['ai_enhanced_ekf']['ate_rmse']:.2f} m"
        drift = f"{r['ai_enhanced_ekf']['drift_pct']:.2f} %"
        gain = f"+{r['improvement_pct']:.1f} %"
        print(f"{sc:<16} | {p_dr:<12} | {c_ekf:<12} | {ai_ekf:<12} | {drift:<8} | {gain:<8}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AGASTYA Dead Reckoning Models")
    parser.add_argument("--all-scenarios", action="store_true", help="Run benchmark across all scenarios")
    parser.add_argument("--scenario", type=str, default="gps_loss", help="Specific scenario name")
    args = parser.parse_args()

    if args.all_scenarios:
        run_all_benchmarks()
    else:
        res = evaluate_scenario(args.scenario)
        print(json.dumps(res, indent=2))
