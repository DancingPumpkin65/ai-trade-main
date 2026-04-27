interface MetricCardProps {
  label: string;
  value: string;
  accent?: string;
}

export function MetricCard({ label, value, accent }: MetricCardProps) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong style={accent ? { color: accent } : undefined}>{value}</strong>
    </article>
  );
}
