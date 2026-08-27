import React from 'react';
import { TelemetryFrame } from '../../types/navigation';
import { Compass, Navigation } from 'lucide-react';

interface NavigationStateProps {
  currentFrame: TelemetryFrame | null;
}

export const NavigationState: React.FC<NavigationStateProps> = ({ currentFrame }) => {
  const euler = currentFrame?.estimated.euler || { roll: 0, pitch: 0, yaw: 0 };
  const pos = currentFrame?.estimated.position || [0, 0, 0];
  const vel = currentFrame?.estimated.velocity || [0, 0, 0];

  const speed = Math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2);
  const altitude = -pos[2]; // In NED, Down is positive, so altitude is -Down

  // Artificial horizon pitch offset (pixels per degree)
  const pitchOffset = Math.max(-60, Math.min(60, euler.pitch * 2.5));
  const rollAngle = euler.roll;

  return (
    <div className="avionics-card p-4 space-y-4 text-slate-200">
      <div className="flex items-center justify-between border-b border-cyan-500/20 pb-2">
        <div className="flex items-center gap-2 font-hud text-sm font-bold tracking-wider text-cyan-400">
          <Navigation className="w-4 h-4 text-cyan-400" />
          PRIMARY FLIGHT ATTITUDE (PFD)
        </div>
        <span className="font-mono-tech text-xs px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30 text-cyan-300">
          {currentFrame?.mode.toUpperCase() || 'STANDBY'}
        </span>
      </div>

      {/* Artificial Horizon Instrument */}
      <div className="relative w-full h-48 bg-[#0a1224] rounded-lg overflow-hidden border border-cyan-500/30 shadow-inner">
        {/* Sky / Ground Rotating Horizon Background */}
        <div
          className="absolute inset-[-50%] transition-transform duration-75 ease-out"
          style={{
            transform: `translate(0px, ${pitchOffset}px) rotate(${-rollAngle}deg)`,
          }}
        >
          {/* Sky (Blue gradient) */}
          <div className="w-full h-1/2 bg-gradient-to-t from-[#1e40af]/80 to-[#0284c7]/90 border-b border-white" />
          {/* Ground (Brown / Dark Ground) */}
          <div className="w-full h-1/2 bg-gradient-to-b from-[#78350f]/90 to-[#451a03]/90" />

          {/* Pitch Ladder Marks */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-[10px] font-mono-tech font-bold text-white/90">
            <div className="space-y-4">
              <div className="flex items-center gap-2"><span>+20</span><div className="w-8 h-[1.5px] bg-white" /><span>+20</span></div>
              <div className="flex items-center gap-2"><span>+10</span><div className="w-12 h-[1.5px] bg-white" /><span>+10</span></div>
              <div className="w-20 h-[2px] bg-emerald-400 shadow-[0_0_8px_#00ff88]" />
              <div className="flex items-center gap-2"><span>-10</span><div className="w-12 h-[1.5px] bg-white border-dashed" /><span>-10</span></div>
              <div className="flex items-center gap-2"><span>-20</span><div className="w-8 h-[1.5px] bg-white border-dashed" /><span>-20</span></div>
            </div>
          </div>
        </div>

        {/* Static Center Aircraft Crosshair */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-10 h-10 border-2 border-amber-400/80 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
          </div>
          <div className="absolute w-28 h-[2px] bg-amber-400/90 shadow-[0_0_8px_#ffb700]" />
        </div>

        {/* Top Roll Pointer Arc */}
        <div className="absolute top-1 left-1/2 -translate-x-1/2 flex flex-col items-center">
          <div className="w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[8px] border-t-amber-400" />
          <span className="text-[10px] font-mono-tech font-bold text-amber-300">
            {rollAngle.toFixed(1)}°
          </span>
        </div>

        {/* Speed Tape (Left) */}
        <div className="absolute left-2 top-1/2 -translate-y-1/2 bg-[#060913]/85 border border-cyan-500/40 p-1.5 rounded text-center">
          <div className="text-[9px] text-slate-400 uppercase font-mono-tech">IAS</div>
          <div className="text-sm font-bold font-mono-tech text-cyan-300 glow-cyan">
            {speed.toFixed(1)}
          </div>
          <div className="text-[9px] text-cyan-500">m/s</div>
        </div>

        {/* Altitude Tape (Right) */}
        <div className="absolute right-2 top-1/2 -translate-y-1/2 bg-[#060913]/85 border border-cyan-500/40 p-1.5 rounded text-center">
          <div className="text-[9px] text-slate-400 uppercase font-mono-tech">ALT</div>
          <div className="text-sm font-bold font-mono-tech text-emerald-300 glow-green">
            {altitude.toFixed(1)}
          </div>
          <div className="text-[9px] text-emerald-500">m</div>
        </div>
      </div>

      {/* 360-Degree Heading Compass Ribbon */}
      <div className="bg-[#070c18] border border-cyan-500/20 p-2.5 rounded-lg">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className="flex items-center gap-1 font-mono-tech text-cyan-400 font-semibold">
            <Compass className="w-3.5 h-3.5" /> HDG
          </span>
          <span className="font-mono-tech text-cyan-300 font-bold text-sm">
            {Math.round(euler.yaw)}° {getCardinalHeading(euler.yaw)}
          </span>
        </div>

        <div className="relative h-6 bg-[#04060d] rounded border border-cyan-500/20 overflow-hidden flex items-center justify-center">
          <div className="absolute w-1 h-full bg-cyan-400 z-10" />
          <div
            className="flex gap-6 font-mono-tech text-xs text-slate-400 transition-transform duration-75"
            style={{
              transform: `translateX(${-((euler.yaw % 360) * 2)}px)`,
            }}
          >
            {[-180, -90, 0, 90, 180, 270, 360, 450, 540].map((deg, i) => (
              <span key={i} className="whitespace-nowrap font-bold">
                {deg % 360 === 0 ? 'N' : deg % 360 === 90 ? 'E' : deg % 360 === 180 ? 'S' : deg % 360 === 270 ? 'W' : `${((deg % 360) + 360) % 360}°`}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Numerical Coordinate Readouts */}
      <div className="grid grid-cols-3 gap-2 text-xs font-mono-tech">
        <div className="bg-[#070d1a] border border-cyan-500/15 p-2 rounded">
          <div className="text-[10px] text-slate-400">NORTH (X)</div>
          <div className="text-cyan-300 font-semibold">{pos[0].toFixed(2)} m</div>
        </div>
        <div className="bg-[#070d1a] border border-cyan-500/15 p-2 rounded">
          <div className="text-[10px] text-slate-400">EAST (Y)</div>
          <div className="text-cyan-300 font-semibold">{pos[1].toFixed(2)} m</div>
        </div>
        <div className="bg-[#070d1a] border border-cyan-500/15 p-2 rounded">
          <div className="text-[10px] text-slate-400">DOWN (Z)</div>
          <div className="text-cyan-300 font-semibold">{pos[2].toFixed(2)} m</div>
        </div>
      </div>
    </div>
  );
};

function getCardinalHeading(yawDeg: number): string {
  const norm = ((yawDeg % 360) + 360) % 360;
  if (norm >= 337.5 || norm < 22.5) return 'N';
  if (norm >= 22.5 && norm < 67.5) return 'NE';
  if (norm >= 67.5 && norm < 112.5) return 'E';
  if (norm >= 112.5 && norm < 157.5) return 'SE';
  if (norm >= 157.5 && norm < 202.5) return 'S';
  if (norm >= 202.5 && norm < 247.5) return 'SW';
  if (norm >= 247.5 && norm < 292.5) return 'W';
  return 'NW';
}
