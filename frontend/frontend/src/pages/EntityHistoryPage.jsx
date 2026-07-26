import { useCallback } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, ReferenceArea } from "recharts";
import { useApi } from "../hooks/useApi";
import { getEntityHistory } from "../api/client";
import RiskBadge from "../components/common/RiskBadge";
import { EmptyBlock, ErrorBlock, LoadingBlock } from "../components/common/StateBlocks";
import { entityTypeLabel, formatRelativeShort, formatTimestamp } from "../utils/format";

export default function EntityHistoryPage() {
  const { entityId } = useParams();
  const navigate = useNavigate();
  const fetcher = useCallback(() => getEntityHistory(entityId, { limit: 200 }), [entityId]);
  const { data: entity, loading, error, reload } = useApi(fetcher, [entityId]);

  if (loading) return <LoadingBlock label={`Loading history for ${entityId}\u2026`} />;
  if (error) return <ErrorBlock error={error} onRetry={reload} />;
  if (!entity) return null;

  const chartData = [...entity.events].reverse().map((e) => ({
    ts: e.timestamp,
    label: formatRelativeShort(e.timestamp),
    risk_score: e.risk_score,
  }));

  return (
    <div className="page">
      <div className="breadcrumb">
        <Link to="/alerts">Alert Queue</Link>
        <ChevronRight size={14} />
        <span className="mono">{entity.entity_id}</span>
      </div>

      <div className="page-header">
        <div>
          <h1 style={{ fontSize: "var(--fs-xl)" }} className="mono">
            {entity.entity_id}
          </h1>
          <p style={{ color: "var(--text-secondary)", marginTop: 4 }}>
            {entityTypeLabel(entity.entity_type)}
            {entity.department && ` \u00b7 ${entity.department}`}
            {entity.role && ` \u00b7 ${entity.role}`}
          </p>
        </div>
      </div>

      <div className="stat-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        <SummaryCard label="Total events" value={entity.total_events.toLocaleString()} />
        <SummaryCard label="Peak risk score" value={entity.max_risk_score.toFixed(1)} accent="var(--risk-critical)" />
        <SummaryCard label="Labeled attack events" value={entity.attack_event_count.toLocaleString()} accent="var(--risk-high)" />
      </div>

      <section className="panel panel--padded">
        <div className="panel__header">
          <h2 style={{ fontSize: "var(--fs-md)" }}>Risk score over time</h2>
          <span className="eyebrow">Most recent {entity.events.length} events</span>
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <ReferenceArea y1={80} y2={100} fill="var(--risk-critical)" fillOpacity={0.08} />
            <ReferenceArea y1={60} y2={80} fill="var(--risk-high)" fillOpacity={0.06} />
            <XAxis dataKey="label" tick={{ fill: "var(--text-tertiary)", fontSize: 10 }} axisLine={{ stroke: "var(--border-subtle)" }} tickLine={false} minTickGap={30} />
            <YAxis domain={[0, 100]} tick={{ fill: "var(--text-tertiary)", fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
            <Tooltip
              labelFormatter={(_, payload) => (payload?.[0] ? formatTimestamp(payload[0].payload.ts) : "")}
              contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 12 }}
            />
            <Line type="monotone" dataKey="risk_score" stroke="var(--accent-info)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <div className="panel__header" style={{ padding: "var(--sp-4) var(--sp-5) 0" }}>
          <h2 style={{ fontSize: "var(--fs-md)" }}>Event history</h2>
        </div>
        {entity.events.length === 0 ? (
          <EmptyBlock title="No events recorded" />
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Risk</th>
                <th>Resource accessed</th>
                <th>Predicted attack</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {entity.events.map((event) => (
                <tr key={event.log_id} onClick={() => navigate(`/alerts/${event.log_id}`)}>
                  <td>
                    <RiskBadge level={event.risk_level} size="sm" />
                    <span className="mono" style={{ marginLeft: 8, fontSize: "var(--fs-xs)", color: "var(--text-secondary)" }}>
                      {event.risk_score.toFixed(1)}
                    </span>
                  </td>
                  <td>{event.resource_accessed || "\u2014"}</td>
                  <td>
                    {event.predicted_attack_type ? (
                      event.predicted_attack_type.replace(/_/g, " ")
                    ) : (
                      <span style={{ color: "var(--text-tertiary)" }}>None</span>
                    )}
                  </td>
                  <td className="mono" style={{ color: "var(--text-secondary)" }}>
                    {formatTimestamp(event.timestamp)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

function SummaryCard({ label, value, accent }) {
  return (
    <div className="panel panel--padded">
      <div className="eyebrow" style={{ marginBottom: "var(--sp-2)" }}>
        {label}
      </div>
      <div className="mono" style={{ fontSize: "var(--fs-xl)", fontWeight: 600, color: accent || "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}
