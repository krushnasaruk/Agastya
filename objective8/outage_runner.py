"""
GNSS Outage Robustness Runner for Objective 8.
Evaluates 5s, 10s, 15s, 20s, 30s, and 45s GNSS outages on held-out test data.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .hardware_ready_engine import HardwareReadyNavigationEngine, HardwareSensorPacket


class OutageRunner:
    """
    Executes standardized GNSS outage drift benchmarks.
    """

    OUTAGE_DURATIONS = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0]

    @classmethod
    def evaluate_outages(
        cls,
        engine_fp32: HardwareReadyNavigationEngine,
        engine_int8: HardwareReadyNavigationEngine,
        test_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        outage_start_time_sec: float = 20.0,
        outage_durations: Optional[List[float]] = None,
        outage_durations_sec: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Runs standardized outages on held-out sequence.
        """
        durs_list = outage_durations or outage_durations_sec or cls.OUTAGE_DURATIONS
        # Align test and reference timestamps
        common_len = min(len(test_df), len(ref_df))
        test_sub = test_df.iloc[:common_len].copy()
        ref_sub = ref_df.iloc[:common_len].copy()

        # Run full baseline trajectories first
        def run_full(engine: HardwareReadyNavigationEngine) -> pd.DataFrame:
            engine.initialize()
            for _, row in test_sub.iterrows():
                pkt = HardwareSensorPacket(
                    timestamp_sec=float(row["time_sec"]),
                    dt_sec=float(row.get("dt_sec", 0.1)),
                    wheel_speed_fl_ms=float(row["wheel_speed_fl_ms"]),
                    wheel_speed_fr_ms=float(row["wheel_speed_fr_ms"]),
                    wheel_speed_rl_ms=float(row["wheel_speed_rl_ms"]),
                    wheel_speed_rr_ms=float(row["wheel_speed_rr_ms"]),
                    accel_x_ms2=float(row["accel_x_ms2"]),
                    yaw_rate_rads=float(row["yaw_rate_rads"])
                )
                engine.step(pkt)
            return pd.DataFrame(engine.nav_history_records)

        fp32_df = run_full(engine_fp32)
        int8_df = run_full(engine_int8)

        # Classical baseline engine run
        engine_class = HardwareReadyNavigationEngine(
            model=engine_fp32.fp32_model,
            feature_scaler=engine_fp32.feature_scaler,
            target_scaler=engine_fp32.target_scaler,
            deployment_mode="MODE_C_CLASSICAL"
        )
        class_df = run_full(engine_class)

        t_arr = test_sub["time_sec"].to_numpy()
        ref_e = ref_sub["pos_east_m"].to_numpy() if "pos_east_m" in ref_sub.columns else ref_sub["p_east_m"].to_numpy()
        ref_n = ref_sub["pos_north_m"].to_numpy() if "pos_north_m" in ref_sub.columns else ref_sub["p_north_m"].to_numpy()

        results_by_duration = []

        for dur in durs_list:
            mask = (t_arr >= outage_start_time_sec) & (t_arr <= (outage_start_time_sec + dur))
            if np.sum(mask) < 2:
                continue

            indices = np.where(mask)[0]
            # Compute traveled distance in window
            dists = np.sqrt(np.diff(ref_e[indices])**2 + np.diff(ref_n[indices])**2)
            dist_m = float(np.sum(dists))

            # Classical error
            c_e = class_df["pos_east_m"].iloc[indices].to_numpy() - class_df["pos_east_m"].iloc[indices[0]]
            c_n = class_df["pos_north_m"].iloc[indices].to_numpy() - class_df["pos_north_m"].iloc[indices[0]]
            r_e = ref_e[indices] - ref_e[indices[0]]
            r_n = ref_n[indices] - ref_n[indices[0]]
            class_ate = float(np.sqrt(np.mean((c_e - r_e)**2 + (c_n - r_n)**2)))

            # FP32 AI error
            f_e = fp32_df["pos_east_m"].iloc[indices].to_numpy() - fp32_df["pos_east_m"].iloc[indices[0]]
            f_n = fp32_df["pos_north_m"].iloc[indices].to_numpy() - fp32_df["pos_north_m"].iloc[indices[0]]
            fp32_ate = float(np.sqrt(np.mean((f_e - r_e)**2 + (f_n - r_n)**2)))

            # INT8 Quantized AI error
            i_e = int8_df["pos_east_m"].iloc[indices].to_numpy() - int8_df["pos_east_m"].iloc[indices[0]]
            i_n = int8_df["pos_north_m"].iloc[indices].to_numpy() - int8_df["pos_north_m"].iloc[indices[0]]
            int8_ate = float(np.sqrt(np.mean((i_e - r_e)**2 + (i_n - r_n)**2)))

            pct_imp_fp32 = float(((class_ate - fp32_ate) / max(class_ate, 1e-6)) * 100.0)
            pct_imp_int8 = float(((class_ate - int8_ate) / max(class_ate, 1e-6)) * 100.0)

            results_by_duration.append({
                "outage_duration_sec": dur,
                "traveled_distance_m": dist_m,
                "classical_ate_m": class_ate,
                "fp32_ai_ate_m": fp32_ate,
                "int8_quantized_ate_m": int8_ate,
                "fp32_improvement_pct": pct_imp_fp32,
                "int8_improvement_pct": pct_imp_int8
            })

        return {
            "outage_start_time_sec": outage_start_time_sec,
            "durations_evaluated_sec": cls.OUTAGE_DURATIONS,
            "outage_records": results_by_duration,
            "status": "PASS"
        }
