import { useState, useTransition } from "react";

import type { AlpacaOrderIntent, TradeOpportunityList, UniverseScanCandidateRecord } from "../lib/types";
import { formatCurrency, formatPercent } from "../lib/utils";
import { StatusBadge } from "./StatusBadge";

interface OpportunityBoardProps {
  requestId: string;
  opportunityList: TradeOpportunityList;
  candidates: UniverseScanCandidateRecord[];
  opportunityOrders: Record<string, AlpacaOrderIntent>;
  onLoadOrder: (symbol: string) => Promise<void>;
  onApproveOrder: (symbol: string) => Promise<void>;
  onRejectOrder: (symbol: string) => Promise<void>;
}

export function OpportunityBoard({
  requestId,
  opportunityList,
  candidates,
  opportunityOrders,
  onLoadOrder,
  onApproveOrder,
  onRejectOrder,
}: OpportunityBoardProps) {
  const [busySymbol, setBusySymbol] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function run(symbol: string, action: (targetSymbol: string) => Promise<void>) {
    setError(null);
    setBusySymbol(symbol);
    startTransition(() => {
      void action(symbol)
        .catch((nextError) => {
          setError(nextError instanceof Error ? nextError.message : "Opportunity order action failed.");
        })
        .finally(() => {
          setBusySymbol(null);
        });
    });
  }

  return (
    <div className="stack">
      <section className="panel">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Universe Scan</p>
            <h2>Top opportunities</h2>
          </div>
        </div>

        <div className="opportunity-grid">
          {opportunityList.top_opportunities.map((opportunity) => (
            <article key={`${opportunity.signal.symbol}-${opportunity.rank}`} className="opportunity-card">
              <div className="opportunity-head">
                <div>
                  <p className="eyebrow">Rank {opportunity.rank}</p>
                  <h3>{opportunity.signal.symbol}</h3>
                </div>
                <StatusBadge value={opportunity.signal.action} />
              </div>
              <p className="support-copy">{opportunity.signal.rationale_fr}</p>
              <dl className="mini-metrics">
                <div>
                  <dt>Confidence</dt>
                  <dd>{formatPercent(opportunity.signal.confidence)}</dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{formatCurrency(opportunity.signal.position_value_mad)}</dd>
                </div>
                <div>
                  <dt>Risk score</dt>
                  <dd>{opportunity.signal.risk_score.toFixed(2)}</dd>
                </div>
              </dl>
              <div className="opportunity-order-box">
                <div className="opportunity-order-head">
                  <strong>Broker command</strong>
                  {opportunityOrders[opportunity.signal.symbol] ? (
                    <StatusBadge value={opportunityOrders[opportunity.signal.symbol].status} />
                  ) : (
                    <span className="order-placeholder">Not prepared</span>
                  )}
                </div>
                {opportunityOrders[opportunity.signal.symbol]?.reason ? (
                  <p className="support-copy">
                    {opportunityOrders[opportunity.signal.symbol].reason}
                  </p>
                ) : null}
                <div className="action-row">
                  {!opportunityOrders[opportunity.signal.symbol] ? (
                    <button
                      className="ghost-button"
                      disabled={isPending && busySymbol === opportunity.signal.symbol}
                      onClick={() => run(opportunity.signal.symbol, onLoadOrder)}
                      type="button"
                    >
                      Prepare preview
                    </button>
                  ) : null}
                  {opportunityOrders[opportunity.signal.symbol]?.status === "PREPARED" ? (
                    <>
                      <button
                        className="primary-button"
                        disabled={isPending && busySymbol === opportunity.signal.symbol}
                        onClick={() => run(opportunity.signal.symbol, onApproveOrder)}
                        type="button"
                      >
                        Approve
                      </button>
                      <button
                        className="danger-button"
                        disabled={isPending && busySymbol === opportunity.signal.symbol}
                        onClick={() => run(opportunity.signal.symbol, onRejectOrder)}
                        type="button"
                      >
                        Reject
                      </button>
                    </>
                  ) : null}
                </div>
              </div>
            </article>
          ))}
        </div>

        {error ? <p className="inline-error">{error}</p> : null}
        <p className="micro-copy">Universe request: {requestId}</p>

        {opportunityList.rejected_candidates_summary.length > 0 ? (
          <div className="alert-strip">
            <strong>Rejected candidates</strong>
            <ul className="compact-list">
              {opportunityList.rejected_candidates_summary.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Ranking Store</p>
            <h2>Candidate audit trail</h2>
          </div>
        </div>

        <div className="table-shell">
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Status</th>
                <th>Score</th>
                <th>Deep eval</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((candidate) => (
                <tr key={`${candidate.symbol}-${candidate.evaluation_status}`}>
                  <td>{candidate.symbol}</td>
                  <td>
                    <StatusBadge value={candidate.evaluation_status.toUpperCase()} />
                  </td>
                  <td>{candidate.score?.toFixed(3) ?? "—"}</td>
                  <td>{candidate.selected_for_deep_eval ? "Yes" : "No"}</td>
                  <td>{candidate.rejection_reason ?? candidate.reasons.join(" · ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
