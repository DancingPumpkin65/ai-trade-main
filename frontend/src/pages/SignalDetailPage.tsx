import { useCallback, useEffect, useState, useTransition } from "react";
import { Link, useParams } from "react-router-dom";

import { EventTimeline } from "../components/EventTimeline";
import { MetricCard } from "../components/MetricCard";
import { OpportunityBoard } from "../components/OpportunityBoard";
import { OrderReviewCard } from "../components/OrderReviewCard";
import { StatusBadge } from "../components/StatusBadge";
import { api } from "../lib/api";
import type { SignalDetailResponse, SignalEvent } from "../lib/types";
import { enumLabel, formatCurrency, formatDateTime, formatPercent } from "../lib/utils";

export function SignalDetailPage() {
  const { requestId = "" } = useParams();
  const [detail, setDetail] = useState<SignalDetailResponse | null>(null);
  const [events, setEvents] = useState<SignalEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, startTransition] = useTransition();

  const load = useCallback(async () => {
    const payload = await api.signalDetail(requestId);
    setDetail(payload);
  }, [requestId]);

  useEffect(() => {
    if (!requestId) {
      return;
    }
    setError(null);
    setEvents([]);
    void load().catch((loadError: unknown) => {
      setError(loadError instanceof Error ? loadError.message : "Unable to load request detail.");
    });

    const controller = new AbortController();
    void api.replaySignalEvents(
      requestId,
      (event) => {
        setEvents((current) => [...current, event]);
      },
      controller.signal,
    ).catch((streamError) => {
      if (controller.signal.aborted) {
        return;
      }
      setError(streamError instanceof Error ? streamError.message : "Unable to replay stored events.");
    });

    return () => controller.abort();
  }, [load, requestId]);

  function refresh() {
    setError(null);
    startTransition(() => {
      void load().catch((loadError: unknown) => {
        setError(loadError instanceof Error ? loadError.message : "Unable to refresh request.");
      });
    });
  }

  async function approve() {
    await api.approveOrder(requestId);
    await load();
  }

  async function reject() {
    await api.rejectOrder(requestId);
    await load();
  }

  if (!requestId) {
    return (
      <section className="panel empty-panel">
        <p>Missing request id.</p>
      </section>
    );
  }

  if (!detail && !error) {
    return (
      <section className="panel empty-panel">
        <p>Loading request detail…</p>
      </section>
    );
  }

  return (
    <div className="stack detail-stack">
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Signal Detail</p>
            <h2>{detail?.final_signal?.symbol ?? detail?.request_intent.symbols_requested[0] ?? "Universe scan"}</h2>
          </div>
          <div className="action-row">
            <StatusBadge value={detail?.signal_status ?? "FAILED"} />
            <button className="ghost-button" onClick={refresh} type="button">
              {isRefreshing ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>

        <div className="detail-grid">
          <div>
            <dt>Request id</dt>
            <dd>{requestId}</dd>
          </div>
          <div>
            <dt>Request mode</dt>
            <dd>{detail?.request_intent.request_mode ?? "—"}</dd>
          </div>
          <div>
            <dt>Risk profile</dt>
            <dd>{detail?.request_intent.risk_preference ?? "—"}</dd>
          </div>
          <div>
            <dt>Horizon</dt>
            <dd>{detail?.request_intent.time_horizon ?? "—"}</dd>
          </div>
          <div>
            <dt>Capital</dt>
            <dd>{formatCurrency(detail?.request_intent.capital_mad)}</dd>
          </div>
          <div>
            <dt>Parser confidence</dt>
            <dd>{formatPercent(detail?.request_intent.parser_confidence)}</dd>
          </div>
        </div>

        <div className="callout">
          <strong>Operator note</strong>
          <p>{detail?.request_intent.operator_visible_note_fr ?? "—"}</p>
        </div>

        {detail?.request_intent.raw_prompt ? (
          <div className="callout muted">
            <strong>Original prompt</strong>
            <p>{detail.request_intent.raw_prompt}</p>
          </div>
        ) : null}

        {detail?.analysis_warnings.length ? (
          <div className="alert-strip danger">
            <strong>Analysis warnings</strong>
            <ul className="compact-list">
              {detail.analysis_warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {error ? <p className="inline-error">{error}</p> : null}
      </section>

      {detail?.final_signal ? (
        <section className="panel panel-dark">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow eyebrow-dark">Final Signal</p>
              <h2>
                {detail.final_signal.symbol} · {enumLabel(detail.final_signal.action)}
              </h2>
            </div>
            <div className="action-row">
              <StatusBadge value={detail.coordinator_output?.intent_alignment ?? "ALIGNED"} />
              <StatusBadge value={detail.alpaca_order_status} />
            </div>
          </div>

          <div className="metric-row">
            <MetricCard label="Position value" value={formatCurrency(detail.final_signal.position_value_mad)} />
            <MetricCard label="Position size" value={formatPercent(detail.final_signal.position_size_pct)} />
            <MetricCard label="Stop loss" value={formatPercent(detail.final_signal.stop_loss_pct)} />
            <MetricCard label="Take profit" value={formatPercent(detail.final_signal.take_profit_pct)} />
            <MetricCard label="Confidence" value={formatPercent(detail.final_signal.confidence)} />
            <MetricCard label="Generated at" value={formatDateTime(detail.final_signal.generated_at)} />
          </div>

          <div className="callout">
            <strong>French rationale</strong>
            <p>{detail.final_signal.rationale_fr}</p>
          </div>

          {detail.final_signal.execution_warnings.length ? (
            <div className="alert-strip">
              <strong>Execution warnings</strong>
              <ul className="compact-list">
                {detail.final_signal.execution_warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </section>
      ) : null}

      {detail?.opportunity_list ? (
        <OpportunityBoard
          candidates={detail.universe_scan_candidates}
          opportunityList={detail.opportunity_list}
        />
      ) : null}

      <OrderReviewCard
        status={detail?.alpaca_order_status ?? "NOT_PREPARED"}
        order={detail?.alpaca_order ?? null}
        onApprove={approve}
        onReject={reject}
      />

      <EventTimeline events={events} />

      <p className="back-link">
        <Link to="/">Back to dashboard</Link>
      </p>
    </div>
  );
}
