const tickerItems = [
  { label: "MASI", value: "+0.84%", tone: "up" },
  { label: "ATW", value: "+1.12%", tone: "up" },
  { label: "BCP", value: "-0.46%", tone: "down" },
  { label: "IAM", value: "+0.21%", tone: "up" },
  { label: "Order Gate", value: "Prepared only", tone: "neutral" },
];

export function PriceTickerStrip() {
  return (
    <div className="ticker-strip" aria-label="Market ticker">
      <div className="ticker-track">
        {tickerItems.map((item) => (
          <div className="ticker-item" key={item.label}>
            <span className="ticker-label">{item.label}</span>
            <strong className={`ticker-value tone-${item.tone}`}>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}
