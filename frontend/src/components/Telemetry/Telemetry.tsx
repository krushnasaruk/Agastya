import React from 'react';
import { TelemetryFrame } from '../../types/navigation';
import { BarChart3, TrendingDown, Target, Zap, Gauge } from 'lucide-react';

interface TelemetryProps {
  currentFrame: TelemetryFrame | null;
}

export const Telemetry: React.FC<TelemetryProps> = ({ currentFrame }) => {
  const metrics = currentFrame?.metrics || {
    ate_rmse: 0.0,
    max_error: 0.0,
    drift_percentage: 0.0,
    total_distance: 0.0,
    ai_confidence: 0.95,
  };

  const progress = (currentFrame?.scenario_progress ?? 0) * 100;

  return (
    <div className="avionics-card p-4 space-y-4 text-slate-200">
      <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
        <div className="flex items-center gap-2 font-hud text-sm font-bold tracking-wider text-cyan-400">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          ESTIMATION ACCURACY & ATE METRICS
        </div>
        <span className="font-mono-tech text-xs text-slate-400">
          T: {currentFrame?.timestamp.toFixed(2) || '0.00'}s
        </span>
      </div>

      {/* Scenario Progress Bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs font-mono-tech text-slate-400">
          <span>Scenario Progress ({currentFrame?.scenario_name || 'gps_loss'})</span>
          <span className="text-cyan-300 font-bold">{progress.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-[#04060d] h-2 rounded-full overflow-hidden border border-cyan-500/20">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 via-sky-400 to-emerald-400 transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Key Metric Cards */}
      <div className="grid grid-cols-2 gap-2.5">
        {/* ATE RMSE */}
        <div className="bg-[#070c18] border border-cyan-500/25 p-3 rounded-lg flex items-center gap-3">
          <div className="p-2 rounded bg-cyan-500/20 text-cyan-300">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono-tech text-slate-400">ATE RMSE</div>
            <div className="text-lg font-bold font-mono-tech text-cyan-300 glow-cyan">
              {metrics.ate_rmse.toFixed(2)} <span className="text-xs font-normal text-slate-400">m</span>
            </div>
          </div>
        </div>

        {/* Drift Percentage */}
        <div className="bg-[#070c18] border border-emerald-500/25 p-3 rounded-lg flex items-center gap-3">
          <div className="p-2 rounded bg-emerald-500/20 text-emerald-300">
            <TrendingDown className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono-tech text-slate-400">Drift (% Dist)</div>
            <div className="text-lg font-bold font-mono-tech text-emerald-300 glow-green">
              {metrics.drift_percentage.toFixed(2)} <span className="text-xs font-normal text-slate-400">%</span>
            </div>
          </div>
        </div>

        {/* Max Error */}
        <div className="bg-[#070c18] border border-amber-500/25 p-3 rounded-lg flex items-center gap-3">
          <div className="p-2 rounded bg-amber-500/20 text-amber-300">
            <Gauge className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono-tech text-slate-400">Max Error</div>
            <div className="text-lg font-bold font-mono-tech text-amber-300 glow-amber">
              {metrics.max_error.toFixed(2)} <span className="text-xs font-normal text-slate-400">m</span>
            </div>
          </div>
        </div>

        {/* Total Distance */}
        <div className="bg-[#070c18] border border-cyan-500/25 p-3 rounded-lg flex items-center gap-3">
          <div className="p-2 rounded bg-cyan-500/20 text-cyan-300">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[10px] uppercase font-mono-tech text-slate-400">Distance Travelled</div>
            <div className="text-lg font-bold font-mono-tech text-slate-200">
              {metrics.total_distance.toFixed(1)} <span className="text-xs font-normal text-slate-400">m</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
