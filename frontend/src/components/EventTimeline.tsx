import type { SignalEvent } from "../lib/types";

import { enumLabel } from "../lib/utils";

export function EventTimeline({ events, embedded = false }: { events: SignalEvent[]; embedded?: boolean }) {
  const content = (
    <>
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Event Playback</p>
          <h2>Analysis timeline</h2>
        </div>
      </div>

      {events.length === 0 ? (
        <p className="support-copy">No stored events were replayed for this request yet.</p>
      ) : (
        <ol className="timeline">
          {events.map((event, index) => (
            <li key={`${event.event_type}-${index}`} className="timeline-item">
              <div className="timeline-dot" />
              <div className="timeline-content">
                <div className="timeline-header">
                  <strong>{enumLabel(event.event_type)}</strong>
                </div>
                <pre>{JSON.stringify(event.payload, null, 2)}</pre>
              </div>
            </li>
          ))}
        </ol>
      )}
    </>
  );

  if (embedded) {
    return <div>{content}</div>;
  }

  return (
    <section className="panel">
      {content}
    </section>
  );
}
