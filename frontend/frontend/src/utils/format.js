export const RISK_LEVELS = ["Critical", "High", "Medium", "Low"];

export const ENTITY_TYPES = [
  "user",
  "service_account",
  "edge_device",
  "iot_device",
  "industrial_controller",
  "server",
];

const ENTITY_TYPE_LABELS = {
  user: "User",
  service_account: "Service Account",
  edge_device: "Edge Device",
  iot_device: "IoT Device",
  industrial_controller: "Industrial Controller",
  server: "Server",
};

export function entityTypeLabel(entityType) {
  return ENTITY_TYPE_LABELS[entityType] || entityType;
}

export function riskColorVar(riskLevel) {
  switch (riskLevel) {
    case "Critical":
      return "var(--risk-critical)";
    case "High":
      return "var(--risk-high)";
    case "Medium":
      return "var(--risk-medium)";
    case "Low":
      return "var(--risk-low)";
    default:
      return "var(--text-tertiary)";
  }
}

export function formatTimestamp(ts) {
  if (!ts) return "\u2014";
  const d = new Date(ts.replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatRelativeShort(ts) {
  if (!ts) return "\u2014";
  const d = new Date(ts.replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return "\u2014";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value, digits = 1) {
  if (value === null || value === undefined) return "\u2014";
  return Number(value).toFixed(digits);
}

export function truncateId(id, length = 8) {
  if (!id) return "\u2014";
  return id.length > length ? `${id.slice(0, length)}\u2026` : id;
}
