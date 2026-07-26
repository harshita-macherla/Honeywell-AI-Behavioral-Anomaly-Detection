import { Bar, BarChart, Cell, ResponsiveContainer, ReferenceLine, Tooltip, XAxis, YAxis } from "recharts";

/**
 * ShapChart
 * =========
 * Visualizes the `top_shap_contributions` array returned by
 * GET /api/v1/explain/{log_id} (or POST /api/v1/predict, indirectly, via
 * its `reasons`) as a signed tornado-style horizontal bar chart -- the
 * standard way to present SHAP attributions: bars pointing right push the
 * model TOWARD the predicted attack type, bars pointing left push away
 * from it. Sign and magnitude both come straight from the backend's SHAP
 * TreeExplainer output; nothing is recomputed or approximated here.
 */

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="panel panel--padded" style={{ padding: "var(--sp-3)", maxWidth: 260 }}>
      <p style={{ fontWeight: 600, marginBottom: 4 }}>{row.readable_name}</p>
      <p className="mono" style={{ fontSize: "var(--fs-xs)", color: "var(--text-secondary)" }}>
        feature: {row.feature}
      </p>
      <p className="mono" style={{ fontSize: "var(--fs-sm)", marginTop: 4, color: row.shap_value >= 0 ? "var(--risk-critical)" : "var(--accent-info)" }}>
        SHAP value: {row.shap_value >= 0 ? "+" : ""}
        {row.shap_value.toFixed(4)}
      </p>
    </div>
  );
}

export default function ShapChart({ contributions }) {
  if (!contributions || contributions.length === 0) return null;

  const data = [...contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 10);
  const chartHeight = Math.max(240, data.length * 34);

  return (
    <div>
      <div style={{ display: "flex", gap: "var(--sp-4)", marginBottom: "var(--sp-3)", fontSize: "var(--fs-xs)", color: "var(--text-secondary)" }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--risk-critical)", display: "inline-block" }} />
          Pushes toward prediction
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: "var(--accent-info)", display: "inline-block" }} />
          Pushes away
        </span>
      </div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fill: "var(--text-tertiary)", fontSize: 11 }} axisLine={{ stroke: "var(--border-subtle)" }} tickLine={false} />
          <YAxis
            type="category"
            dataKey="readable_name"
            width={220}
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            axisLine={{ stroke: "var(--border-subtle)" }}
            tickLine={false}
          />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--bg-panel-raised)" }} />
          <Bar dataKey="shap_value" radius={[3, 3, 3, 3]} maxBarSize={16}>
            {data.map((entry) => (
              <Cell key={entry.feature} fill={entry.shap_value >= 0 ? "var(--risk-critical)" : "var(--accent-info)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
