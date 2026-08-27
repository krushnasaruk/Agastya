# AGASTYA Cyber-Avionics Mission Control Dashboard

## Overview
A modern, high-performance React 18 + Vite + TypeScript telemetry and mission control dashboard for AI Dead Reckoning navigation systems.

## Features
- **Real-Time Multi-Track Trajectory Canvas**: 60 FPS multi-layer rendering of Ground Truth (Green), AI-Enhanced ES-EKF (Cyan), Classical EKF (Blue), Pure Dead Reckoning (Crimson), and Raw GNSS points (Amber) with 3-sigma covariance ellipses.
- **Primary Flight Display (PFD)**: 3D Artificial Horizon, Roll Angle Pointer, Speed/Altitude Tapes, 360° Compass Ribbon.
- **Multi-Sensor Oscilloscope**: Live 6-DOF IMU acceleration and gyro waveforms, GNSS satellite geometry, and Visual Odometry tracking confidence meters.
- **Interactive Fault Injection**: Dynamic buttons to trigger GPS Jamming blackouts, accelerometer bias shifts, and optical tracking loss.
- **Scenario Selector & Speed Controls**: Instant switching between `normal`, `gps_loss`, `gps_noise`, and `urban_canyon` scenarios with 1x/2x/5x playback speeds.

## Quick Start
```bash
npm install
npm run dev
```
Visit `http://localhost:5173`.
