"""
15-State Error-State Extended Kalman Filter (ES-EKF).
Fuses high-rate IMU Strapdown Mechanization with asynchronous GNSS,
Visual Odometry, AI Neural Velocity inferences, and Zero-Velocity updates (ZUPT).
Includes Joseph-Form covariance propagation and condition number monitoring.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .state import (
    NavigationState,
    quat_normalize,
    quat_multiply,
    quat_to_rotation_matrix,
    skew_symmetric
)
from .dead_reckoning import StrapdownDeadReckoning


class ErrorStateKalmanFilter:
    def __init__(
        self,
        accel_noise_density: float = 0.05,     # m/s^2 / sqrt(Hz)
        gyro_noise_density: float = 0.005,     # rad/s / sqrt(Hz)
        accel_bias_random_walk: float = 0.001, # m/s^3 / sqrt(Hz)
        gyro_bias_random_walk: float = 0.0001, # rad/s^2 / sqrt(Hz)
        gravity_accel: float = 9.80665
    ):
        self.sins = StrapdownDeadReckoning(gravity_accel=gravity_accel)
        self.gravity = gravity_accel

        # Continuous Noise Power Spectral Densities
        self.q_a = (accel_noise_density) ** 2
        self.q_g = (gyro_noise_density) ** 2
        self.q_ba = (accel_bias_random_walk) ** 2
        self.q_bg = (gyro_bias_random_walk) ** 2

    def get_condition_number(self, state: NavigationState) -> float:
        """Computes the 2-norm condition number of the state covariance matrix."""
        try:
            return float(np.linalg.cond(state.covariance))
        except Exception:
            return float("inf")

    def is_covariance_healthy(self, state: NavigationState, max_condition: float = 1e12) -> bool:
        """Validates that covariance matrix is symmetric, positive semi-definite, and well-conditioned."""
        P = state.covariance
        if not np.all(np.isfinite(P)):
            return False
        # Check symmetry
        if not np.allclose(P, P.T, atol=1e-6):
            return False
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(P)
        if np.any(eigenvalues < -1e-7):
            return False
        # Check condition number
        cond = self.get_condition_number(state)
        return cond < max_condition

    def predict(
        self,
        state: NavigationState,
        accel_meas: np.ndarray,
        gyro_meas: np.ndarray,
        dt: float
    ) -> NavigationState:
        """
        1. Propagate nominal state via SINS mechanization.
        2. Compute linearized continuous system matrix F.
        3. Form discrete state transition matrix Phi = exp(F * dt) approx I + F*dt.
        4. Propagate 15x15 covariance P = Phi * P * Phi^T + Q_d.
        """
        # Step 1: Nominal state propagation
        nominal_state = self.sins.step(state, accel_meas, gyro_meas, dt)

        # Step 2: System matrix F (15x15)
        # States: [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]
        F = np.zeros((15, 15), dtype=np.float64)
        
        # dp_dot = dv
        F[0:3, 3:6] = np.eye(3)

        # dv_dot = -[f_n]x * dtheta - C_b_n * dba
        C_b_n = state.get_rotation_matrix()
        f_b = accel_meas - state.accel_bias
        f_n = C_b_n @ f_b
        F[3:6, 6:9] = -skew_symmetric(f_n)
        F[3:6, 9:12] = -C_b_n

        # dtheta_dot = -C_b_n * dbg
        F[6:9, 12:15] = -C_b_n

        # Step 3: Discrete Transition Matrix Phi = I + F*dt + 0.5*F^2*dt^2
        I15 = np.eye(15, dtype=np.float64)
        Phi = I15 + F * dt + 0.5 * (F @ F) * (dt ** 2)

        # Step 4: Discrete Process Noise Covariance Q_d
        Q_c = np.zeros((12, 12), dtype=np.float64)
        Q_c[0:3, 0:3] = np.eye(3) * self.q_a
        Q_c[3:6, 3:6] = np.eye(3) * self.q_g
        Q_c[6:9, 6:9] = np.eye(3) * self.q_ba
        Q_c[9:12, 9:12] = np.eye(3) * self.q_bg

        # Noise mapping matrix G (15x12)
        G = np.zeros((15, 12), dtype=np.float64)
        G[3:6, 0:3] = -C_b_n
        G[6:9, 3:6] = -C_b_n
        G[9:12, 6:9] = np.eye(3)
        G[12:15, 9:12] = np.eye(3)

        Q_d = G @ Q_c @ G.T * dt

        # Covariance propagation
        P_pred = Phi @ state.covariance @ Phi.T + Q_d
        
        # Enforce symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)
        nominal_state.covariance = P_pred

        return nominal_state

    def update_gnss_pva(
        self,
        state: NavigationState,
        gnss_pos_ned: np.ndarray,
        gnss_vel_ned: np.ndarray,
        R_pos: np.ndarray,
        R_vel: np.ndarray,
        max_mahalanobis_sq: float = 300.0
    ) -> Tuple[NavigationState, float]:
        """
        Measurement update using 6-DOF GNSS Position & Velocity.
        z = [p_gnss - p_pred, v_gnss - v_pred]^T (6x1)
        """
        H = np.zeros((6, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)  # dp
        H[3:6, 3:6] = np.eye(3)  # dv

        R = np.zeros((6, 6), dtype=np.float64)
        R[0:3, 0:3] = R_pos
        R[3:6, 3:6] = R_vel

        y = np.zeros(6, dtype=np.float64)
        y[0:3] = gnss_pos_ned - state.position
        y[3:6] = gnss_vel_ned - state.velocity

        return self._apply_update(state, y, H, R, max_mahalanobis_sq)

    def update_velocity(
        self,
        state: NavigationState,
        vel_meas_ned: np.ndarray,
        R_vel: np.ndarray,
        max_mahalanobis_sq: float = 2000.0
    ) -> Tuple[NavigationState, float]:
        """
        Measurement update using 3D Velocity (from AI inference, VO, or ZUPT).
        z = v_meas - v_pred (3x1)
        """
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)

        y = vel_meas_ned - state.velocity
        return self._apply_update(state, y, H, R_vel, max_mahalanobis_sq)

    def update_zero_velocity(
        self,
        state: NavigationState,
        R_zupt: Optional[np.ndarray] = None,
        max_mahalanobis_sq: float = 500.0
    ) -> Tuple[NavigationState, float]:
        """
        Direct Zero Velocity Update (ZUPT) enforcing stationary velocity constraint.
        """
        if R_zupt is None:
            R_zupt = np.eye(3, dtype=np.float64) * 1e-4
        return self.update_velocity(state, np.zeros(3, dtype=np.float64), R_zupt, max_mahalanobis_sq)

    def _apply_update(
        self,
        state: NavigationState,
        y: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        max_mahalanobis_sq: float
    ) -> Tuple[NavigationState, float]:
        """
        General Kalman Measurement Update with Joseph-Form covariance
        and Indirect Error State Injection.
        """
        P = state.covariance
        S = H @ P @ H.T + R
        
        # Numerical conditioning of S
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Mahalanobis gating for outlier rejection
        mahalanobis_sq = float(y.T @ S_inv @ y)
        if mahalanobis_sq > max_mahalanobis_sq:
            # Reject outlier
            return state.clone(), mahalanobis_sq

        # Kalman Gain K (15 x m)
        K = P @ H.T @ S_inv

        # Error State Correction dx (15,)
        dx = K @ y

        # Joseph-form covariance update: P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I15 = np.eye(15, dtype=np.float64)
        IKH = I15 - K @ H
        P_updated = IKH @ P @ IKH.T + K @ R @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)

        # Indirect State Injection
        updated_state = state.clone()
        updated_state.covariance = P_updated

        # 1. Position & Velocity injection
        updated_state.position += dx[0:3]
        updated_state.velocity += dx[3:6]

        # 2. Bias injection with realistic slow decay
        updated_state.accel_bias += dx[9:12] * 0.01
        updated_state.gyro_bias += dx[12:15] * 0.001

        return updated_state, mahalanobis_sq
