import React, { useEffect, useRef, useState } from 'react';
import { TelemetryFrame } from '../../types/navigation';
import { ZoomIn, ZoomOut, RotateCcw, Crosshair, Layers, Eye, EyeOff } from 'lucide-react';

interface MapViewProps {
  currentFrame: TelemetryFrame | null;
}

interface Point2D {
  x: number; // East (m)
  y: number; // North (m)
}

export const MapView: React.FC<MapViewProps> = ({ currentFrame }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Trajectory history buffers
  const gtTrail = useRef<Point2D[]>([]);
  const aiTrail = useRef<Point2D[]>([]);
  const classicTrail = useRef<Point2D[]>([]);
  const pureDrTrail = useRef<Point2D[]>([]);
  const gnssPoints = useRef<Point2D[]>([]);

  // Canvas Viewport Transformation
  const [scale, setScale] = useState<number>(3.5); // pixels per meter
  const [offset, setOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [autoCenter, setAutoCenter] = useState<boolean>(true);
  const isDragging = useRef<boolean>(false);
  const lastMousePos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Layer visibility toggles
  const [showGT, setShowGT] = useState(true);
  const [showAI, setShowAI] = useState(true);
  const [showClassic, setShowClassic] = useState(true);
  const [showPureDR, setShowPureDR] = useState(true);
  const [showGNSS, setShowGNSS] = useState(true);
  const [showCovariance, setShowCovariance] = useState(true);

  // Update trails when new frame arrives
  useEffect(() => {
    if (!currentFrame) return;

    const gt = currentFrame.ground_truth.position;
    const est = currentFrame.estimated.position;
    const cl = currentFrame.classical_ekf.position;
    const dr = currentFrame.pure_dr.position;

    // In NED: North is y (up on map), East is x (right on map)
    gtTrail.current.push({ x: gt[1], y: gt[0] });
    aiTrail.current.push({ x: est[1], y: est[0] });
    classicTrail.current.push({ x: cl[1], y: cl[0] });
    pureDrTrail.current.push({ x: dr[1], y: dr[0] });

    if (currentFrame.gnss && currentFrame.gnss.is_valid) {
      gnssPoints.current.push({ x: currentFrame.gnss.position[1], y: currentFrame.gnss.position[0] });
      if (gnssPoints.current.length > 200) gnssPoints.current.shift();
    }

    // Limit buffer length
    const MAX_TRAIL = 1200;
    if (gtTrail.current.length > MAX_TRAIL) gtTrail.current.shift();
    if (aiTrail.current.length > MAX_TRAIL) aiTrail.current.shift();
    if (classicTrail.current.length > MAX_TRAIL) classicTrail.current.shift();
    if (pureDrTrail.current.length > MAX_TRAIL) pureDrTrail.current.shift();

    // Auto-center on current vehicle position
    if (autoCenter && canvasRef.current) {
      const cw = canvasRef.current.width;
      const ch = canvasRef.current.height;
      setOffset({
        x: cw / 2 - est[1] * scale,
        y: ch / 2 + est[0] * scale,
      });
    }
  }, [currentFrame, autoCenter, scale]);

  // Main Canvas Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI displays
    const width = canvas.parentElement?.clientWidth || 800;
    const height = canvas.parentElement?.clientHeight || 600;
    canvas.width = width;
    canvas.height = height;

    // Background Clear
    ctx.fillStyle = '#060913';
    ctx.fillRect(0, 0, width, height);

    // Draw Coordinate Grid
    const gridSizeMeters = 20; // 20m grid
    const gridPixelSize = gridSizeMeters * scale;

    ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
    ctx.lineWidth = 1;

    const startX = offset.x % gridPixelSize;
    const startY = offset.y % gridPixelSize;

    for (let x = startX; x < width; x += gridPixelSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = startY; y < height; y += gridPixelSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Origin Axes (0,0)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, offset.y);
    ctx.lineTo(width, offset.y);
    ctx.moveTo(offset.x, 0);
    ctx.lineTo(offset.x, height);
    ctx.stroke();

    // Helper: World (East, North) -> Canvas (X, Y)
    const toCanvas = (pt: Point2D) => ({
      x: offset.x + pt.x * scale,
      y: offset.y - pt.y * scale, // Invert Y because North is up
    });

    // 1. Draw Raw GNSS Scatter Points
    if (showGNSS && gnssPoints.current.length > 0) {
      ctx.fillStyle = 'rgba(255, 183, 0, 0.5)';
      gnssPoints.current.forEach((pt) => {
        const c = toCanvas(pt);
        ctx.beginPath();
        ctx.arc(c.x, c.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // 2. Draw Pure Dead Reckoning Trail (Crimson)
    if (showPureDR && pureDrTrail.current.length > 1) {
      ctx.strokeStyle = 'rgba(255, 42, 95, 0.75)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      pureDrTrail.current.forEach((pt, idx) => {
        const c = toCanvas(pt);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 3. Draw Classical EKF Trail (Sky Blue)
    if (showClassic && classicTrail.current.length > 1) {
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      classicTrail.current.forEach((pt, idx) => {
        const c = toCanvas(pt);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
    }

    // 4. Draw Ground Truth Trail (Radar Green)
    if (showGT && gtTrail.current.length > 1) {
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(0, 255, 136, 0.5)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      gtTrail.current.forEach((pt, idx) => {
        const c = toCanvas(pt);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0; // Reset shadow
    }

    // 5. Draw AI-Enhanced ES-EKF Trail (Neon Cyan)
    if (showAI && aiTrail.current.length > 1) {
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 3;
      ctx.shadowColor = 'rgba(0, 240, 255, 0.7)';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      aiTrail.current.forEach((pt, idx) => {
        const c = toCanvas(pt);
        if (idx === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 6. Draw 3-Sigma Covariance Error Ellipse around Estimated Pos
    if (showCovariance && currentFrame) {
      const est = currentFrame.estimated.position;
      const cov = currentFrame.estimated.cov_diag;
      const sigmaN = Math.sqrt(cov[0] || 0.05) * 3 * scale;
      const sigmaE = Math.sqrt(cov[1] || 0.05) * 3 * scale;

      const cEst = toCanvas({ x: est[1], y: est[0] });

      ctx.save();
      ctx.translate(cEst.x, cEst.y);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(sigmaE, 6), Math.max(sigmaN, 6), 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.12)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // 7. Draw Current Vehicle Marker & Heading
    if (currentFrame) {
      const est = currentFrame.estimated.position;
      const yawRad = (currentFrame.estimated.euler.yaw * Math.PI) / 180;
      const c = toCanvas({ x: est[1], y: est[0] });

      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.rotate(yawRad); // In 2D canvas, positive angle rotates clockwise

      // Vehicle Triangular Glyph
      ctx.fillStyle = '#00f0ff';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, -14); // Nose
      ctx.lineTo(10, 10); // Right wing
      ctx.lineTo(0, 6);   // Tail center
      ctx.lineTo(-10, 10); // Left wing
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Heading Vector Line
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, -14);
      ctx.lineTo(0, -40);
      ctx.stroke();

      ctx.restore();
    }
  }, [currentFrame, offset, scale, showGT, showAI, showClassic, showPureDR, showGNSS, showCovariance]);

  // Mouse Interaction handlers (Pan & Drag)
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    lastMousePos.current = { x: e.clientX, y: e.clientY };
    setAutoCenter(false);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - lastMousePos.current.x;
    const dy = e.clientY - lastMousePos.current.y;
    lastMousePos.current = { x: e.clientX, y: e.clientY };
    setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => Math.max(0.5, Math.min(prev * zoomFactor, 25.0)));
  };

  const resetView = () => {
    setScale(3.5);
    setAutoCenter(true);
  };

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-lg overflow-hidden border border-cyan-500/20 bg-[#060913]">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Trajectory Legend Overlay */}
      <div className="absolute top-3 left-3 bg-[#0a101e]/85 backdrop-blur-md border border-cyan-500/20 p-2.5 rounded-md text-xs space-y-1.5 shadow-lg">
        <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider font-hud flex items-center gap-1.5 mb-1">
          <Layers className="w-3.5 h-3.5" /> Trajectory Layers
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowGT(!showGT)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#00ff88] shadow-[0_0_6px_#00ff88]" />
            <span>Ground Truth</span>
          </button>
          {showGT ? <Eye className="w-3 h-3 text-emerald-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowAI(!showAI)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#00f0ff] shadow-[0_0_6px_#00f0ff]" />
            <span>AI-Enhanced EKF</span>
          </button>
          {showAI ? <Eye className="w-3 h-3 text-cyan-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowClassic(!showClassic)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#38bdf8]" />
            <span>Classical EKF</span>
          </button>
          {showClassic ? <Eye className="w-3 h-3 text-sky-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowPureDR(!showPureDR)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff2a5f] shadow-[0_0_6px_#ff2a5f]" />
            <span>Pure Dead Reckoning</span>
          </button>
          {showPureDR ? <Eye className="w-3 h-3 text-rose-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowGNSS(!showGNSS)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-[#ffb700]" />
            <span>Raw GNSS Fixes</span>
          </button>
          {showGNSS ? <Eye className="w-3 h-3 text-amber-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>

        <div className="flex items-center justify-between gap-3">
          <button
            onClick={() => setShowCovariance(!showCovariance)}
            className="flex items-center gap-2 text-slate-300 hover:text-white"
          >
            <span className="w-2.5 h-2.5 rounded-full border border-cyan-400 bg-cyan-500/20" />
            <span>3-Sigma Ellipses</span>
          </button>
          {showCovariance ? <Eye className="w-3 h-3 text-cyan-400" /> : <EyeOff className="w-3 h-3 text-slate-500" />}
        </div>
      </div>

      {/* Interactive Controls Overlay */}
      <div className="absolute top-3 right-3 flex flex-col gap-1.5 bg-[#0a101e]/85 backdrop-blur-md border border-cyan-500/20 p-1.5 rounded-md shadow-lg">
        <button
          onClick={() => setScale((s) => Math.min(s * 1.25, 25.0))}
          className="p-1.5 hover:bg-cyan-500/20 text-cyan-300 rounded transition"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setScale((s) => Math.max(s * 0.8, 0.5))}
          className="p-1.5 hover:bg-cyan-500/20 text-cyan-300 rounded transition"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => setAutoCenter(!autoCenter)}
          className={`p-1.5 rounded transition ${autoCenter ? 'bg-cyan-500/30 text-cyan-200' : 'hover:bg-cyan-500/20 text-slate-400'}`}
          title="Auto Center on Vehicle"
        >
          <Crosshair className="w-4 h-4" />
        </button>
        <button
          onClick={resetView}
          className="p-1.5 hover:bg-cyan-500/20 text-cyan-300 rounded transition"
          title="Reset Zoom & Pan"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Scale & Grid Footer */}
      <div className="absolute bottom-2 left-3 text-[11px] font-mono-tech text-cyan-400/80 bg-[#060913]/80 px-2 py-0.5 rounded border border-cyan-500/15">
        Grid: 20m | Zoom: {(scale * 10).toFixed(0)}% | Auto-Center: {autoCenter ? 'LOCKED' : 'FREE'}
      </div>
    </div>
  );
};
