import { Link } from "react-router-dom";

import type { SignalRecord } from "../lib/types";
import { formatDateTime } from "../lib/utils";
import { StatusBadge } from "./StatusBadge";

export function HistoryTable({ records }: { records: SignalRecord[] }) {
  if (records.length === 0) {
    return (
      <section className="panel empty-panel">
        <h2>No analysis runs yet</h2>
        <p>Use the request composer to start the first signal or universe scan.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">History</p>
          <h2>Recent requests</h2>
        </div>
      </div>

      <div className="table-shell">
        <table className="data-table">
          <thead>
            <tr>
              <th>Request</th>
              <th>Mode</th>
              <th>Status</th>
              <th>Alpaca</th>
              <th>Prompt</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr key={record.request_id}>
                <td>
                  <Link to={`/signals/${record.request_id}`} className="table-link">
                    <strong>{record.request_intent.symbols_requested[0] ?? "Universe scan"}</strong>
                    <span>{record.request_id.slice(0, 8)}</span>
                  </Link>
                </td>
                <td>{record.request_intent.request_mode}</td>
                <td>
                  <StatusBadge value={record.status} />
                </td>
                <td>
                  <StatusBadge value={record.alpaca_order_status} />
                </td>
                <td>
                  <div className="table-prompt">
                    <span>{record.request_intent.raw_prompt ?? "Structured request"}</span>
                    <small>{formatDateTime(record.final_signal?.generated_at ?? null)}</small>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
