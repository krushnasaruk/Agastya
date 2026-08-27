export type NavigationMode = 'ai_enhanced_ekf' | 'classical_ekf' | 'pure_dr' | 'ai_only';

export interface EulerAngles {
  roll: number;
  pitch: number;
  yaw: number;
}

export interface NavigationState {
  timestamp: number;
  position: [number, number, number];
  velocity: [number, number, number];
  quaternion: [number, number, number, number];
  euler: EulerAngles;
  accel_bias: [number, number, number];
  gyro_bias: [number, number, number];
  cov_diag: number[];
  mode: string;
  gnss_valid: boolean;
}

export interface GroundTruthState {
  timestamp: number;
  position: [number, number, number];
  velocity: [number, number, number];
  euler: EulerAngles;
}

export interface IMUData {
  timestamp: number;
  accel: [number, number, number];
  gyro: [number, number, number];
  temperature: number;
  is_valid: boolean;
}

export interface GNSSData {
  timestamp: number;
  position: [number, number, number];
  velocity: [number, number, number];
  fix_type: number;
  satellites_in_view: number;
  hdop: number;
  vdop: number;
  cov_diag: number[];
  is_valid: boolean;
}

export interface VOData {
  timestamp: number;
  velocity_body: [number, number, number];
  confidence: number;
  inlier_count: number;
  is_valid: boolean;
}

export interface TelemetryMetrics {
  ate_rmse: number;
  max_error: number;
  drift_percentage: number;
  total_distance: number;
  ai_confidence: number;
}

export interface TelemetryFrame {
  timestamp: number;
  mode: string;
  scenario_name: string;
  scenario_progress: number;
  ground_truth: GroundTruthState;
  estimated: NavigationState;
  pure_dr: NavigationState;
  classical_ekf: NavigationState;
  imu: IMUData;
  gnss: GNSSData | null;
  vo: VOData | null;
  gnss_available: boolean;
  metrics: TelemetryMetrics;
}

export interface ScenarioInfo {
  id: string;
  name: string;
  description: string;
  duration_sec: number;
  trajectory_type: string;
  speed_mps: number;
}
