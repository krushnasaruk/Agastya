import React, { useEffect, useState } from 'react';
import { TelemetryFrame, ScenarioInfo, NavigationMode } from '../../types/navigation';
import {
  telemetryClient,
  fetchScenarios,
  selectScenario,
  startSimulation,
  pauseSimulation,
  resetSimulation,
  setSimulationSpeed,
  setNavigationMode,
  injectFault,
} from '../../services/api';
import { MapView } from '../../components/MapView/MapView';
import { NavigationState } from '../../components/NavigationState/NavigationState';
import { SensorStatus } from '../../components/SensorStatus/SensorStatus';
import { Telemetry } from '../../components/Telemetry/Telemetry';
import {
  Play,
  Pause,
  RotateCcw,
  Sliders,
  ShieldAlert,
  Zap,
  Radio,
  EyeOff,
  Cpu,
  Flame,
  CheckCircle2,
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const [currentFrame, setCurrentFrame] = useState<TelemetryFrame | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [activeScenario, setActiveScenario] = useState<string>('gps_loss');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1.0);
  const [currentMode, setCurrentMode] = useState<NavigationMode>('ai_enhanced_ekf');
  const [activeFault, setActiveFault] = useState<string | null>(null);

  // Subscribe to real-time 50Hz WebSocket stream
  useEffect(() => {
    telemetryClient.connect();
    const unsubscribe = telemetryClient.subscribe((frame) => {
      setCurrentFrame(frame);
    });

    // Load available scenarios
    fetchScenarios()
      .then((data) => setScenarios(data))
      .catch((err) => console.error('Failed to load scenarios:', err));

    return () => {
      unsubscribe();
      telemetryClient.disconnect();
    };
  }, []);

  const handleScenarioChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const scId = e.target.value;
    setActiveScenario(scId);
    await selectScenario(scId);
  };

  const handlePlayPause = async () => {
    if (isPlaying) {
      await pauseSimulation();
      setIsPlaying(false);
    } else {
      await startSimulation();
      setIsPlaying(true);
    }
  };

  const handleReset = async () => {
    await resetSimulation();
  };

  const handleSpeedChange = async (speed: number) => {
    setSimSpeed(speed);
    await setSimulationSpeed(speed);
  };

  const handleModeSelect = async (mode: NavigationMode) => {
    setCurrentMode(mode);
    await setNavigationMode(mode);
  };

  const handleFaultTrigger = async (faultType: string, val: number = 1.0) => {
    setActiveFault(faultType);
    await injectFault(faultType, val);
    setTimeout(() => setActiveFault(null), 3000);
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#060913] text-slate-100 p-3 gap-3 overflow-hidden cyber-grid">
      {/* Top Header & Mission Control Bar */}
      <header className="avionics-card px-4 py-2.5 flex items-center justify-between shadow-lg">
        {/* Branding & Status */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center shadow-[0_0_12px_#00f0ff]">
            <Zap className="w-5 h-5 text-slate-950 fill-current" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-hud font-bold text-base tracking-wider text-cyan-400 glow-cyan">
                AGASTYA
              </h1>
              <span className="text-[10px] font-mono-tech px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30 text-cyan-300">
                AI DEAD RECKONING ENGINE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono-tech">
              Strapdown SINS • 15-State ES-EKF • BiLSTM Neural Inertial Navigation
            </p>
          </div>
        </div>

        {/* Playback & Scenario Controls */}
        <div className="flex items-center gap-3">
          {/* Scenario Selector */}
          <div className="flex items-center gap-2 bg-[#04060d] px-3 py-1.5 rounded-md border border-cyan-500/25">
            <span className="text-xs text-slate-400 font-mono-tech">Scenario:</span>
            <select
              value={activeScenario}
              onChange={handleScenarioChange}
              className="bg-transparent text-cyan-300 font-mono-tech text-xs font-semibold focus:outline-none cursor-pointer"
            >
              {scenarios.map((sc) => (
                <option key={sc.id} value={sc.id} className="bg-[#0b1220] text-slate-200">
                  {sc.name.toUpperCase()} ({sc.duration_sec}s)
                </option>
              ))}
            </select>
          </div>

          {/* Play/Pause & Reset */}
          <div className="flex items-center gap-1.5 bg-[#04060d] p-1 rounded-md border border-cyan-500/25">
            <button
              onClick={handlePlayPause}
              className={`p-1.5 rounded transition ${isPlaying ? 'bg-amber-500/20 text-amber-300 hover:bg-amber-500/30' : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'}`}
              title={isPlaying ? 'Pause Simulation' : 'Resume Simulation'}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={handleReset}
              className="p-1.5 rounded hover:bg-cyan-500/20 text-cyan-300 transition"
              title="Reset Simulation Clock"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Playback Speed Multipliers */}
          <div className="flex items-center bg-[#04060d] p-1 rounded-md border border-cyan-500/25 text-xs font-mono-tech">
            {[1.0, 2.0, 5.0].map((s) => (
              <button
                key={s}
                onClick={() => handleSpeedChange(s)}
                className={`px-2 py-1 rounded transition ${simSpeed === s ? 'bg-cyan-500/30 text-cyan-200 font-bold' : 'text-slate-400 hover:text-white'}`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <div className="grid grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left Column: Attitude Horizon + Multi-Sensor Status */}
        <div className="col-span-3 flex flex-col gap-3 overflow-y-auto">
          <NavigationState currentFrame={currentFrame} />
          <SensorStatus currentFrame={currentFrame} />
        </div>

        {/* Center Column: High-Precision Map Visualizer */}
        <div className="col-span-6 flex flex-col gap-3 min-h-0">
          <div className="flex-1 min-h-0">
            <MapView currentFrame={currentFrame} />
          </div>

          {/* Navigation Operating Mode Bar */}
          <div className="avionics-card px-3 py-2 flex items-center justify-between">
            <span className="text-xs font-hud font-bold text-cyan-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" /> FUSION MODE
            </span>
            <div className="flex gap-2 text-xs font-mono-tech">
              {(
                [
                  { id: 'ai_enhanced_ekf', label: 'AI-Enhanced ES-EKF' },
                  { id: 'classical_ekf', label: 'Classical EKF' },
                  { id: 'pure_dr', label: 'Pure Dead Reckoning' },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  onClick={() => handleModeSelect(m.id)}
                  className={`px-2.5 py-1 rounded border transition ${
                    currentMode === m.id
                      ? 'bg-cyan-500/25 border-cyan-400 text-cyan-200 font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                      : 'border-cyan-500/15 text-slate-400 hover:text-white hover:border-cyan-500/30'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Telemetry Metrics + Fault Injection Suite */}
        <div className="col-span-3 flex flex-col gap-3 overflow-y-auto">
          <Telemetry currentFrame={currentFrame} />

          {/* Fault Injection Panel */}
          <div className="avionics-card p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
              <div className="flex items-center gap-2 font-hud text-sm font-bold tracking-wider text-rose-400">
                <ShieldAlert className="w-4 h-4 text-rose-400" />
                SENSOR FAULT INJECTION
              </div>
            </div>

            <p className="text-xs text-slate-400 font-mono-tech">
              Test resilience against EW electronic jamming, thermal bias jumps, and sensor occlusion.
            </p>

            <div className="space-y-2">
              <button
                onClick={() => handleFaultTrigger('gps_jamming', 1.0)}
                className={`w-full py-2 px-3 rounded text-xs font-mono-tech flex items-center justify-between border transition ${
                  activeFault === 'gps_jamming'
                    ? 'bg-rose-500/30 border-rose-500 text-rose-200 font-bold shadow-[0_0_12px_#ff2a5f]'
                    : 'bg-[#070c18] border-rose-500/20 text-rose-300 hover:bg-rose-950/40 hover:border-rose-500/40'
                }`}
              >
                <span className="flex items-center gap-2">
                  <Radio className="w-3.5 h-3.5" /> Toggle GPS Jamming Blackout
                </span>
                <Flame className="w-3.5 h-3.5 text-rose-400" />
              </button>

              <button
                onClick={() => handleFaultTrigger('accel_bias_jump', 0.5)}
                className={`w-full py-2 px-3 rounded text-xs font-mono-tech flex items-center justify-between border transition ${
                  activeFault === 'accel_bias_jump'
                    ? 'bg-amber-500/30 border-amber-500 text-amber-200 font-bold'
                    : 'bg-[#070c18] border-amber-500/20 text-amber-300 hover:bg-amber-950/40'
                }`}
              >
                <span className="flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5" /> Inject Accelerometer Bias Jump (+0.5 m/s²)
                </span>
                <Zap className="w-3.5 h-3.5 text-amber-400" />
              </button>

              <button
                onClick={() => handleFaultTrigger('gyro_bias_jump', 0.05)}
                className={`w-full py-2 px-3 rounded text-xs font-mono-tech flex items-center justify-between border transition ${
                  activeFault === 'gyro_bias_jump'
                    ? 'bg-amber-500/30 border-amber-500 text-amber-200 font-bold'
                    : 'bg-[#070c18] border-amber-500/20 text-amber-300 hover:bg-amber-950/40'
                }`}
              >
                <span className="flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5" /> Inject Gyroscope Drift (+0.05 rad/s)
                </span>
                <Zap className="w-3.5 h-3.5 text-amber-400" />
              </button>

              <button
                onClick={() => handleFaultTrigger('vo_dropout', 1.0)}
                className={`w-full py-2 px-3 rounded text-xs font-mono-tech flex items-center justify-between border transition ${
                  activeFault === 'vo_dropout'
                    ? 'bg-sky-500/30 border-sky-500 text-sky-200 font-bold'
                    : 'bg-[#070c18] border-sky-500/20 text-sky-300 hover:bg-sky-950/40'
                }`}
              >
                <span className="flex items-center gap-2">
                  <EyeOff className="w-3.5 h-3.5" /> Simulate Optical Tracking Loss
                </span>
                <CheckCircle2 className="w-3.5 h-3.5 text-sky-400" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
