import type {
  AlpacaOrderEnvelope,
  AuthResponse,
  GenerateSignalRequest,
  GenerateSignalResponse,
  HealthResponse,
  SignalDetailResponse,
  SignalEvent,
  SignalRecord,
} from "./types";
import { getStoredToken } from "./tokenStorage";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getStoredToken();
  const headers = new Headers(init?.headers ?? {});
  headers.set("Content-Type", "application/json");
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      throw new Error(payload.detail ?? fallback);
    } catch (error) {
      if (error instanceof Error && error.message !== fallback) {
        throw error;
      }
      throw new Error(fallback);
    }
  }

  return (await response.json()) as T;
}

export const api = {
  health() {
    return request<HealthResponse>("/health", { method: "GET" });
  },
  history() {
    return request<SignalRecord[]>("/history", { method: "GET" });
  },
  generateSignal(payload: GenerateSignalRequest) {
    return request<GenerateSignalResponse>("/signals/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  signalDetail(requestId: string) {
    return request<SignalDetailResponse>(`/signals/${requestId}`, { method: "GET" });
  },
  approveOrder(requestId: string) {
    return request<SignalRecord>(`/signals/${requestId}/approve`, { method: "POST" });
  },
  rejectOrder(requestId: string) {
    return request<SignalRecord>(`/signals/${requestId}/reject`, { method: "POST" });
  },
  alpacaOrder(requestId: string) {
    return request<AlpacaOrderEnvelope>(`/signals/${requestId}/alpaca-order`, { method: "GET" });
  },
  register(username: string, password: string) {
    return request<{ username: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },
  login(username: string, password: string) {
    return request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
  },
  async replaySignalEvents(
    requestId: string,
    onEvent: (event: SignalEvent) => void,
    signal?: AbortSignal,
  ) {
    const token = getStoredToken();
    const headers = new Headers();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const response = await fetch(
      `${API_BASE_URL}/signals/generate/stream?request_id=${encodeURIComponent(requestId)}`,
      { headers, signal },
    );
    if (!response.ok || !response.body) {
      throw new Error("Unable to load event stream.");
    }

    const decoder = new TextDecoder();
    const reader = response.body.getReader();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() ?? "";

      for (const chunk of chunks) {
        const lines = chunk.split("\n");
        let eventType = "message";
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data += line.slice(5).trim();
          }
        }
        if (!data) {
          continue;
        }
        onEvent({
          event_type: eventType,
          payload: JSON.parse(data) as Record<string, unknown>,
        });
      }
    }
  },
  async liveGenerateSignal(
    payload: GenerateSignalRequest,
    handlers: {
      onRequestStarted?: (requestId: string) => void;
      onEvent?: (event: SignalEvent) => void;
      signal?: AbortSignal;
    },
  ) {
    const token = getStoredToken();
    const params = new URLSearchParams();
    if (payload.symbol) {
      params.set("symbol", payload.symbol);
    }
    if (payload.capital != null) {
      params.set("capital", String(payload.capital));
    }
    if (payload.prompt) {
      params.set("prompt", payload.prompt);
    }
    if (payload.risk_profile) {
      params.set("risk_profile", payload.risk_profile);
    }
    if (payload.time_horizon) {
      params.set("time_horizon", payload.time_horizon);
    }

    const headers = new Headers();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }

    const response = await fetch(`${API_BASE_URL}/signals/generate/stream?${params.toString()}`, {
      headers,
      signal: handlers.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error("Unable to start live analysis stream.");
    }

    const decoder = new TextDecoder();
    const reader = response.body.getReader();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() ?? "";

      for (const chunk of chunks) {
        const lines = chunk.split("\n");
        let eventType = "message";
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data += line.slice(5).trim();
          }
        }
        if (!data) {
          continue;
        }
        const payloadObject = JSON.parse(data) as Record<string, unknown>;
        if (eventType === "request_started" && typeof payloadObject.request_id === "string") {
          handlers.onRequestStarted?.(payloadObject.request_id);
        }
        handlers.onEvent?.({
          event_type: eventType,
          payload: payloadObject,
        });
      }
    }
  },
};
