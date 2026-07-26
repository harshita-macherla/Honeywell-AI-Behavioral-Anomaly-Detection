/**
 * RiskBadge
 * =========
 * Renders a risk_level string ("Critical" | "High" | "Medium" | "Low") as
 * a color-coded pill. Critical badges carry the signature radar-pulse
 * animation defined in global.css -- reserved for that tier only, so it
 * stays meaningful rather than decorative.
 */
export default function RiskBadge({ level, size = "md" }) {
  const cls = `risk-badge risk-badge--${(level || "").toLowerCase()}`;
  return (
    <span className={cls} style={size === "sm" ? { fontSize: "10px", padding: "2px 8px 2px 6px" } : undefined}>
      <span className="risk-badge__dot" />
      {level || "Unknown"}
    </span>
  );
}
