export default function StatCard({ label, value, accent, icon: Icon, sublabel }) {
  return (
    <div className="panel panel--padded" style={{ display: "flex", flexDirection: "column", gap: "var(--sp-3)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="eyebrow">{label}</span>
        {Icon && <Icon size={16} color={accent || "var(--text-tertiary)"} />}
      </div>
      <span
        className="mono"
        style={{ fontSize: "var(--fs-xxl)", fontWeight: 600, lineHeight: 1, color: accent || "var(--text-primary)" }}
      >
        {value}
      </span>
      {sublabel && <span style={{ fontSize: "var(--fs-sm)", color: "var(--text-secondary)" }}>{sublabel}</span>}
    </div>
  );
}
