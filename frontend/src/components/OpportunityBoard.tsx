import type { TradeOpportunityList, UniverseScanCandidateRecord } from "../lib/types";
import { formatCurrency, formatPercent } from "../lib/utils";
import { StatusBadge } from "./StatusBadge";

interface OpportunityBoardProps {
  opportunityList: TradeOpportunityList;
  candidates: UniverseScanCandidateRecord[];
}

export function OpportunityBoard({ opportunityList, candidates }: OpportunityBoardProps) {
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
            </article>
          ))}
        </div>

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
