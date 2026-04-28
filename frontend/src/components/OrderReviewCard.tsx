import { useState, useTransition } from "react";

import type { AlpacaOrderIntent, AlpacaOrderStatus } from "../lib/types";
import { enumLabel, formatCurrency } from "../lib/utils";
import { StatusBadge } from "./StatusBadge";

interface OrderReviewCardProps {
  status: AlpacaOrderStatus;
  order: AlpacaOrderIntent | null;
  onApprove: () => Promise<void>;
  onReject: () => Promise<void>;
}

export function OrderReviewCard({ status, order, onApprove, onReject }: OrderReviewCardProps) {
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function run(action: () => Promise<void>) {
    setError(null);
    startTransition(() => {
      void action().catch((nextError) => {
        setError(nextError instanceof Error ? nextError.message : "Order action failed.");
      });
    });
  }

  return (
    <section className="panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Broker Preview</p>
          <h2>Alpaca command gate</h2>
        </div>
        <StatusBadge value={status} />
      </div>

      {order ? (
        <dl className="detail-grid">
          <div>
            <dt>Source symbol</dt>
            <dd>{order.source_symbol}</dd>
          </div>
          <div>
            <dt>Mapped asset</dt>
            <dd>{order.alpaca_symbol ?? "No mapping"}</dd>
          </div>
          <div>
            <dt>Side</dt>
            <dd>{order.side ? enumLabel(order.side) : "—"}</dd>
          </div>
          <div>
            <dt>Notional</dt>
            <dd>{formatCurrency(order.notional)}</dd>
          </div>
          <div>
            <dt>Submission mode</dt>
            <dd>
              {order.preview_only
                ? "Preview only"
                : `${order.broker_submission_mode ?? "broker"} submit · ${order.broker_order_status ?? "submitted"}`}
            </dd>
          </div>
          <div>
            <dt>Broker order id</dt>
            <dd>{order.broker_order_id ?? "—"}</dd>
          </div>
          <div>
            <dt>Reason</dt>
            <dd>{order.reason ?? "Ready for operator decision."}</dd>
          </div>
        </dl>
      ) : (
        <p className="support-copy">No Alpaca payload is attached to this request.</p>
      )}

      {status === "PREPARED" ? (
        <div className="action-row">
          <button className="primary-button" disabled={isPending} onClick={() => run(onApprove)} type="button">
            {isPending ? "Updating…" : "Approve command"}
          </button>
          <button className="danger-button" disabled={isPending} onClick={() => run(onReject)} type="button">
            Reject command
          </button>
        </div>
      ) : null}

      {error ? <p className="inline-error">{error}</p> : null}
    </section>
  );
}
