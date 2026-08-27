import React, { useEffect, useRef } from 'react';
import { TelemetryFrame } from '../../types/navigation';
import { Activity, Radio, Eye, AlertTriangle, ShieldCheck, Cpu } from 'lucide-react';

interface SensorStatusProps {
  currentFrame: TelemetryFrame | null;
}

export const SensorStatus: React.FC<SensorStatusProps> = ({ currentFrame }) => {
  const imuCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imuHistory = useRef<{ ax: number[]; ay: number[]; az: number[]; gx: number[]; gy: number[]; gz: number[] }>({
    ax: [], ay: [], az: [],
    gx: [], gy: [], gz: []
  });

  const imu = currentFrame?.imu;
  const gnss = currentFrame?.gnss;
  const vo = currentFrame?.vo;
  const isJammed = !currentFrame?.gnss_available;

  // Append IMU waveform history
  useEffect(() => {
    if (!imu) return;
    const MAX_POINTS = 80;

    imuHistory.current.ax.push(imu.accel[0]);
    imuHistory.current.ay.push(imu.accel[1]);
    imuHistory.current.az.push(imu.accel[2]);
    imuHistory.current.gx.push(imu.gyro[0]);
    imuHistory.current.gy.push(imu.gyro[1]);
    imuHistory.current.gz.push(imu.gyro[2]);

    if (imuHistory.current.ax.length > MAX_POINTS) {
      imuHistory.current.ax.shift();
      imuHistory.current.ay.shift();
      imuHistory.current.az.shift();
      imuHistory.current.gx.shift();
      imuHistory.current.gy.shift();
      imuHistory.current.gz.shift();
    }
  }, [imu]);

  // Draw 6-axis IMU Oscilloscope Waveform
  useEffect(() => {
    const canvas = imuCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = canvas.parentElement?.clientWidth || 300;
    const h = canvas.height = 90;

    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, w, h);

    // Center zero lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const drawLine = (data: number[], color: string, scaleY: number) => {
      if (data.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const stepX = w / (data.length - 1);
      data.forEach((val, i) => {
        const x = i * stepX;
        const y = h / 2 - val * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    // Accelerometer X (Red), Y (Green), Z (Blue)
    drawLine(imuHistory.current.ax, '#ff3b69', 2.0);
    drawLine(imuHistory.current.ay, '#10b981', 2.0);
    drawLine(imuHistory.current.az, '#38bdf8', 2.0);
  }, [imu]);

  return (
    <div className="avionics-card p-4 space-y-4 text-slate-200">
      <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
        <div className="flex items-center gap-2 font-hud text-sm font-bold tracking-wider text-cyan-400">
          <Activity className="w-4 h-4 text-cyan-400" />
          MULTI-SENSOR HEALTH & TELEMETRY
        </div>
      </div>

      {/* 6-DOF IMU Oscilloscope */}
      <div className="bg-[#070c18] border border-cyan-500/20 p-2.5 rounded-lg space-y-2">
        <div className="flex items-center justify-between text-xs font-mono-tech">
          <span className="flex items-center gap-1 text-cyan-400 font-semibold">
            <Cpu className="w-3.5 h-3.5" /> 6-DOF IMU (100 Hz)
          </span>
          <span className="text-slate-400 text-[11px]">
            Temp: {imu?.temperature.toFixed(1) || '25.0'}°C
          </span>
        </div>

        <div className="rounded overflow-hidden border border-cyan-500/20 relative">
          <canvas ref={imuCanvasRef} className="w-full h-[90px]" />
          <div className="absolute top-1 left-2 flex gap-3 text-[10px] font-mono-tech">
            <span className="text-[#ff3b69]">Ax: {imu?.accel[0].toFixed(2) || '0.00'}</span>
            <span className="text-[#10b981]">Ay: {imu?.accel[1].toFixed(2) || '0.00'}</span>
            <span className="text-[#38bdf8]">Az: {imu?.accel[2].toFixed(2) || '0.00'}</span>
          </div>
        </div>
      </div>

      {/* GNSS Constellation Status */}
      <div className={`p-2.5 rounded-lg border transition-colors ${isJammed ? 'bg-rose-950/30 border-rose-500/40' : 'bg-[#070c18] border-cyan-500/20'}`}>
        <div className="flex items-center justify-between text-xs font-mono-tech mb-2">
          <span className={`flex items-center gap-1.5 font-semibold ${isJammed ? 'text-rose-400 glow-crimson' : 'text-cyan-400'}`}>
            <Radio className="w-3.5 h-3.5" /> GNSS RECEIVER (5 Hz)
          </span>
          {isJammed ? (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold border border-rose-500/40 text-[10px] animate-pulse">
              <AlertTriangle className="w-3 h-3" /> SIGNAL DENIED / JAMMED
            </span>
          ) : (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30 text-[10px]">
              <ShieldCheck className="w-3 h-3" /> 3D FIX ACTIVE
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-xs font-mono-tech">
          <div className="bg-[#04060d] p-1.5 rounded border border-cyan-500/10">
            <div className="text-[10px] text-slate-400">SATELLITES</div>
            <div className={`font-bold ${isJammed ? 'text-rose-400' : 'text-cyan-300'}`}>
              {gnss?.satellites_in_view ?? (isJammed ? 0 : 12)} SVs
            </div>
          </div>
          <div className="bg-[#04060d] p-1.5 rounded border border-cyan-500/10">
            <div className="text-[10px] text-slate-400">HDOP</div>
            <div className={`font-bold ${isJammed ? 'text-rose-400' : 'text-cyan-300'}`}>
              {gnss?.hdop.toFixed(2) ?? (isJammed ? '99.9' : '1.20')}
            </div>
          </div>
          <div className="bg-[#04060d] p-1.5 rounded border border-cyan-500/10">
            <div className="text-[10px] text-slate-400">STATUS</div>
            <div className={`font-bold ${isJammed ? 'text-rose-400' : 'text-emerald-400'}`}>
              {isJammed ? 'OUTAGE' : 'LOCKED'}
            </div>
          </div>
        </div>
      </div>

      {/* Visual Odometry (Camera) Status */}
      <div className="bg-[#070c18] border border-cyan-500/20 p-2.5 rounded-lg space-y-2">
        <div className="flex items-center justify-between text-xs font-mono-tech">
          <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
            <Eye className="w-3.5 h-3.5" /> VISUAL ODOMETRY (20 Hz)
          </span>
          <span className="text-[11px] text-emerald-400 font-bold">
            {vo?.is_valid ? 'TRACKING' : 'SEARCHING'}
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-[11px] font-mono-tech text-slate-400">
            <span>Feature Confidence</span>
            <span className="text-cyan-300 font-bold">{((vo?.confidence ?? 0.95) * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full bg-[#04060d] h-2 rounded-full overflow-hidden border border-cyan-500/20">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-150"
              style={{ width: `${(vo?.confidence ?? 0.95) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
