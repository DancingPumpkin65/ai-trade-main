export function formatCurrency(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("fr-MA", {
    style: "currency",
    currency: "MAD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number | null | undefined, fractionDigits = 1) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(fractionDigits)}%`;
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) {
    return "—";
  }
  return new Intl.DateTimeFormat("fr-MA", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function enumLabel(value: string) {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function statusTone(value: string) {
  if (value === "COMPLETED" || value === "APPROVED" || value === "ALIGNED") {
    return "success";
  }
  if (value === "PREPARED" || value === "RUNNING" || value === "PARTIALLY_ALIGNED") {
    return "pending";
  }
  if (value === "UNMAPPABLE" || value === "REJECTED" || value === "FAILED" || value === "NOT_ALIGNED") {
    return "danger";
  }
  return "neutral";
}
