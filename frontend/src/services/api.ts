import { NavigationMode, ScenarioInfo, TelemetryFrame } from '../types/navigation';

const API_BASE = '/api';
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/telemetry`;

export async function fetchScenarios(): Promise<ScenarioInfo[]> {
  const res = await fetch(`${API_BASE}/simulation/scenarios`);
  if (!res.ok) throw new Error('Failed to fetch scenarios');
  return res.json();
}

export async function selectScenario(scenarioId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/simulation/scenario/${scenarioId}`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to load scenario ${scenarioId}`);
}

export async function startSimulation(): Promise<void> {
  await fetch(`${API_BASE}/simulation/start`, { method: 'POST' });
}

export async function pauseSimulation(): Promise<void> {
  await fetch(`${API_BASE}/simulation/pause`, { method: 'POST' });
}

export async function resetSimulation(): Promise<void> {
  await fetch(`${API_BASE}/simulation/reset`, { method: 'POST' });
}

export async function setSimulationSpeed(speed: number): Promise<void> {
  await fetch(`${API_BASE}/simulation/speed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed_multiplier: speed }),
  });
}

export async function setNavigationMode(mode: NavigationMode): Promise<void> {
  await fetch(`${API_BASE}/navigation/mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  });
}

export async function injectFault(faultType: string, value: number = 1.0): Promise<void> {
  await fetch(`${API_BASE}/navigation/inject-fault`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fault_type: faultType, value }),
  });
}

export class TelemetryStreamClient {
  private ws: WebSocket | null = null;
  private listeners: Set<(frame: TelemetryFrame) => void> = new Set();
  private isIntentionalClose = false;
  private reconnectTimeout: any = null;

  connect() {
    this.isIntentionalClose = false;
    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        console.log('[TelemetryStream] WebSocket connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const frame: TelemetryFrame = JSON.parse(event.data);
          this.listeners.forEach((listener) => listener(frame));
        } catch (e) {
          console.error('[TelemetryStream] Parsing error:', e);
        }
      };

      this.ws.onclose = () => {
        if (!this.isIntentionalClose) {
          console.warn('[TelemetryStream] Disconnected. Reconnecting in 1.5s...');
          this.reconnectTimeout = setTimeout(() => this.connect(), 1500);
        }
      };

      this.ws.onerror = (err) => {
        console.error('[TelemetryStream] Socket error:', err);
      };
    } catch (e) {
      console.error('[TelemetryStream] Connection failed:', e);
      this.reconnectTimeout = setTimeout(() => this.connect(), 2000);
    }
  }

  subscribe(listener: (frame: TelemetryFrame) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  disconnect() {
    this.isIntentionalClose = true;
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export const telemetryClient = new TelemetryStreamClient();
