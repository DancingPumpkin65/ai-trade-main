import { enumLabel, statusTone } from "../lib/utils";

export function StatusBadge({ value }: { value: string }) {
  const tone = statusTone(value);
  return <span className={`status-badge tone-${tone}`}>{enumLabel(value)}</span>;
}
