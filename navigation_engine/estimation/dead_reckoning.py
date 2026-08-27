"""
Strapdown Inertial Navigation System (SINS) Dead Reckoning Mechanization.
Performs 3D attitude, velocity, and position numerical propagation from IMU specific force
and angular rate readings with gravity compensation.
"""

import numpy as np
from typing import Optional
from .state import (
    NavigationState,
    quat_normalize,
    quat_multiply,
    quat_to_rotation_matrix
)


class StrapdownDeadReckoning:
    def __init__(
        self,
        gravity_accel: float = 9.80665,
        enable_coriolis: bool = False,
        earth_rate_rad_s: float = 7.292115e-5
    ):
        self.g_n = np.array([0.0, 0.0, gravity_accel], dtype=np.float64)
        self.enable_coriolis = enable_coriolis
        self.omega_ie = earth_rate_rad_s

    def propagate_attitude_rk4(
        self,
        q: np.ndarray,
        gyro_meas: np.ndarray,
        gyro_bias: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        4th-Order Runge-Kutta quaternion integration for angular rates.
        q_dot = 0.5 * q * [0, omega]
        """
        omega = gyro_meas - gyro_bias  # Unbiased angular rate (rad/s)
        norm_omega = np.linalg.norm(omega)

        if norm_omega < 1e-10:
            return quat_normalize(q)

        # Closed-form exact rotation quaternion over dt:
        half_angle = 0.5 * norm_omega * dt
        axis = omega / norm_omega
        delta_q = np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ], dtype=np.float64)

        q_next = quat_multiply(q, delta_q)
        return quat_normalize(q_next)

    def step(
        self,
        state: NavigationState,
        accel_meas: np.ndarray,
        gyro_meas: np.ndarray,
        dt: float
    ) -> NavigationState:
        """
        Propagate navigation state forward by dt seconds using SINS mechanization.
        """
        new_state = state.clone()
        new_state.timestamp += dt

        # 1. Attitude Propagation (RK4)
        q_prev = state.quaternion
        q_next = self.propagate_attitude_rk4(q_prev, gyro_meas, state.gyro_bias, dt)
        new_state.quaternion = q_next

        # 2. Midpoint attitude for specific force transformation
        q_mid = quat_normalize(0.5 * (q_prev + q_next))
        C_b_n = quat_to_rotation_matrix(q_mid)

        # 3. Unbiased specific force in navigation frame
        f_b = accel_meas - state.accel_bias
        f_n = C_b_n @ f_b

        # 4. Acceleration in navigation frame (with gravity compensation)
        a_n = f_n + self.g_n

        # Optional Coriolis compensation
        if self.enable_coriolis:
            # Approx for mid-latitudes
            omega_coriolis = np.array([0.0, 0.0, self.omega_ie])
            a_n -= 2.0 * np.cross(omega_coriolis, state.velocity)

        # 5. Velocity and Position Integration (Trapezoidal / Euler)
        new_state.velocity = state.velocity + a_n * dt
        new_state.position = state.position + 0.5 * (state.velocity + new_state.velocity) * dt

        return new_state
