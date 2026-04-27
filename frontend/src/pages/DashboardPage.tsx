import { useEffect, useState, useTransition } from "react";
import { Link, useNavigate } from "react-router-dom";

import { EventTimeline } from "../components/EventTimeline";
import { HistoryTable } from "../components/HistoryTable";
import { MetricCard } from "../components/MetricCard";
import { RequestComposer } from "../components/RequestComposer";
import { StatusBadge } from "../components/StatusBadge";
import { api } from "../lib/api";
import type { GenerateSignalRequest, HealthResponse, SignalEvent, SignalRecord } from "../lib/types";

export function DashboardPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<SignalRecord[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [liveRequestId, setLiveRequestId] = useState<string | null>(null);
  const [liveEvents, setLiveEvents] = useState<SignalEvent[]>([]);
  const [liveStatus, setLiveStatus] = useState<string>("IDLE");
  const [isLiveRunning, setIsLiveRunning] = useState(false);
  const [isRefreshing, startTransition] = useTransition();

  async function loadDashboard() {
    const [historyPayload, healthPayload] = await Promise.all([api.history(), api.health()]);
    setHistory(historyPayload);
    setHealth(healthPayload);
  }

  useEffect(() => {
    void loadDashboard().catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : "Unable to load operator data.");
    });
  }, []);

  async function handleSubmit(payload: GenerateSignalRequest) {
    const response = await api.generateSignal(payload);
    navigate(`/signals/${response.request_id}`);
  }

  async function handleLiveSubmit(payload: GenerateSignalRequest) {
    setLiveError(null);
    setLiveRequestId(null);
    setLiveEvents([]);
    setLiveStatus("RUNNING");
    setIsLiveRunning(true);

    await api.liveGenerateSignal(payload, {
      onRequestStarted(requestId) {
        setLiveRequestId(requestId);
      },
      onEvent(event) {
        setLiveEvents((current) => [...current, event]);
        if (event.event_type === "pipeline_complete") {
          setLiveStatus("COMPLETED");
          setIsLiveRunning(false);
          void loadDashboard().catch(() => undefined);
        }
        if (event.event_type === "pipeline_failed") {
          setLiveStatus("FAILED");
          setIsLiveRunning(false);
        }
      },
    }).catch((streamError) => {
      setIsLiveRunning(false);
      setLiveStatus("FAILED");
      setLiveError(streamError instanceof Error ? streamError.message : "Unable to start live analysis.");
      throw streamError;
    });
  }

  function refresh() {
    setError(null);
    startTransition(() => {
      void loadDashboard().catch((loadError) => {
        setError(loadError instanceof Error ? loadError.message : "Unable to refresh.");
      });
    });
  }

  return (
    <div className="dashboard-stack">
      <section className="hero-surface">
        <div className="hero-copy">
          <p className="eyebrow">Operations Deck</p>
          <h2>Moroccan trade ideas, broker-gated like a real dealing desk.</h2>
          <p className="support-copy hero-text">
            The analysis engine runs to completion, flags risk in plain French, and only stops when a broker command is
            ready to approve, reject, or auto-pass in full-access mode.
          </p>
          <div className="hero-actions">
            <button className="primary-button pill-button" onClick={refresh} type="button">
              {isRefreshing ? "Refreshing…" : "Refresh desk"}
            </button>
            <span className="hero-footnote">Intent parsing, LangGraph execution, and Alpaca preview flow stay visible.</span>
          </div>
        </div>
        <div className="hero-dark-panel">
          <div className="hero-panel-head">
            <p className="eyebrow eyebrow-dark">Desk snapshot</p>
            <span className="signal-dot">Live posture</span>
          </div>
          <div className="metric-row metric-row-tight">
            <MetricCard label="Requests loaded" value={String(history.length)} accent="var(--brand-yellow)" />
            <MetricCard
              label="Approval mode"
              value={health?.alpaca_require_order_approval ? "Operator gated" : "Full access"}
            />
            <MetricCard label="RAG backend" value={health?.rag_backend ?? "—"} />
            <MetricCard label="LangGraph" value={health?.langgraph_enabled ? "Active" : "Fallback"} />
          </div>
        </div>
      </section>

      <div className="dashboard-grid">
        <RequestComposer onSubmit={handleSubmit} onSubmitLive={handleLiveSubmit} />

        {error ? <p className="inline-error field-span-full">{error}</p> : null}

        <section className="panel panel-dark">
          <div className="section-heading compact section-heading-dark">
            <div>
              <p className="eyebrow eyebrow-dark">Service Health</p>
              <h2>Runtime snapshot</h2>
            </div>
          </div>
          <div className="detail-grid">
            <div>
              <dt>Status</dt>
              <dd>{health?.status ?? "Loading"}</dd>
            </div>
            <div>
              <dt>LangGraph</dt>
              <dd>{health?.langgraph_enabled ? "Enabled" : "Fallback runtime"}</dd>
            </div>
            <div>
              <dt>LangSmith tracing</dt>
              <dd>{health?.langsmith_tracing ? "On" : "Off"}</dd>
            </div>
            <div>
              <dt>Alpaca submit</dt>
              <dd>{health?.alpaca_submit_orders ? "Enabled" : "Preview only"}</dd>
            </div>
          </div>
        </section>
      </div>

      <section className="panel panel-dark">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow eyebrow-dark">Live Feed</p>
            <h2>In-progress execution stream</h2>
          </div>
          <div className="action-row">
            <StatusBadge value={liveStatus} />
            {liveRequestId ? (
              <Link className="ghost-button live-link" to={`/signals/${liveRequestId}`}>
                Open detail
              </Link>
            ) : null}
          </div>
        </div>

        <p className="support-copy">
          Launch a live analysis run from the composer to stream graph events here while the backend is still executing.
        </p>

        {isLiveRunning ? <p className="stream-pill">Streaming operator events…</p> : null}
        {liveError ? <p className="inline-error">{liveError}</p> : null}
        {!liveRequestId && !liveEvents.length && !liveError ? (
          <div className="callout">
            <strong>Waiting for a live run</strong>
            <p>The snapshot flow still navigates straight to detail. The live flow keeps this desk open and streams events in place.</p>
          </div>
        ) : (
          <EventTimeline embedded events={liveEvents} />
        )}
      </section>

      <div className="field-span-full">
        <HistoryTable records={history} />
      </div>
    </div>
  );
}
