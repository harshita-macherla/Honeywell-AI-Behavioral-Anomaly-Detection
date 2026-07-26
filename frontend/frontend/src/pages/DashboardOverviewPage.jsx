import { useCallback } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ShieldAlert, Activity, Users, Target, ArrowRight } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { getStatsOverview } from "../api/client";
import StatCard from "../components/common/StatCard";
import { ErrorBlock, LoadingBlock } from "../components/common/StateBlocks";
import { entityTypeLabel, formatPercent, riskColorVar } from "../utils/format";

const RISK_ORDER = ["Critical", "High", "Medium", "Low"];

export default function DashboardOverviewPage() {
  const fetcher = useCallback(() => getStatsOverview(), []);
  const { data: stats, loading, error, reload } = useApi(fetcher, []);

  if (loading) return <LoadingBlock label="Loading dashboard overview\u2026" />;
  if (error) return <ErrorBlock error={error} onRetry={reload} />;
  if (!stats) return null;

  const riskChartData = RISK_ORDER.map((level) => ({
    level,
    count: stats.risk_level_counts[level] ?? 0,
  }));

  const attackTypeData = Object.entries(stats.top_predicted_attack_types)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([type, count]) => ({ type: type.replace(/_/g, " "), count }));

  const maxEntityEvents = Math.max(...stats.entity_type_breakdown.map((e) => e.total_events), 1);

  return (
    <div className="page">
      <PageHeader
        title="Overview"
        subtitle="Live posture across the v2 behavioral-anomaly pipeline"
      />

      <div className="stat-grid">
        <StatCard label="Events monitored" value={stats.total_events.toLocaleString()} icon={Activity} />
        <StatCard label="Entities tracked" value={stats.total_entities.toLocaleString()} icon={Users} />
        <StatCard
          label="Critical alerts"
          value={stats.risk_level_counts.Critical.toLocaleString()}
          accent="var(--risk-critical)"
          icon={ShieldAlert}
        />
        <StatCard
          label="Critical-tier precision"
          value={stats.critical_alert_precision !== null ? formatPercent(stats.critical_alert_precision) : "\u2014"}
          accent="var(--risk-low)"
          icon={Target}
          sublabel="Share of Critical alerts confirmed as true attacks"
        />
      </div>

      <div className="grid-2col">
        <section className="panel panel--padded">
          <div className="panel__header">
            <h2 style={{ fontSize: "var(--fs-md)" }}>Risk distribution</h2>
            <Link to="/alerts" className="panel__header-link">
              View all alerts <ArrowRight size={13} />
            </Link>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={riskChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="level" tick={{ fill: "var(--text-secondary)", fontSize: 12 }} axisLine={{ stroke: "var(--border-subtle)" }} tickLine={false} />
              <YAxis tick={{ fill: "var(--text-tertiary)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: "var(--bg-panel-raised)" }}
                contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 13 }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={64}>
                {riskChartData.map((d) => (
                  <Cell key={d.level} fill={riskColorVar(d.level)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel panel--padded">
          <div className="panel__header">
            <h2 style={{ fontSize: "var(--fs-md)" }}>Top predicted attack types</h2>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={attackTypeData} layout="vertical" margin={{ top: 4, right: 24, left: 4, bottom: 4 }}>
              <XAxis type="number" tick={{ fill: "var(--text-tertiary)", fontSize: 11 }} axisLine={{ stroke: "var(--border-subtle)" }} tickLine={false} />
              <YAxis type="category" dataKey="type" width={150} tick={{ fill: "var(--text-secondary)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: "var(--bg-panel-raised)" }}
                contentStyle={{ background: "var(--bg-panel-raised)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 13 }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={16} fill="var(--accent-info)" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>

      <section className="panel panel--padded" style={{ marginTop: "var(--sp-5)" }}>
        <div className="panel__header">
          <h2 style={{ fontSize: "var(--fs-md)" }}>Entity type coverage</h2>
          <span className="eyebrow">Domain-agnostic: users, service accounts &amp; OT/IoT</span>
        </div>
        <div className="entity-breakdown">
          {stats.entity_type_breakdown.map((row) => (
            <Link to={`/alerts?entity_type=${row.entity_type}`} key={row.entity_type} className="entity-breakdown__row">
              <span className="entity-breakdown__label">{entityTypeLabel(row.entity_type)}</span>
              <div className="entity-breakdown__bar-track">
                <div
                  className="entity-breakdown__bar-fill"
                  style={{ width: `${(row.total_events / maxEntityEvents) * 100}%` }}
                />
              </div>
              <span className="mono entity-breakdown__count">{row.total_events.toLocaleString()}</span>
              <span className="mono entity-breakdown__critical" style={{ color: "var(--risk-critical)" }}>
                {row.critical_alerts} critical
              </span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="page-header">
      <div>
        <h1 style={{ fontSize: "var(--fs-xl)" }}>{title}</h1>
        {subtitle && <p style={{ color: "var(--text-secondary)", marginTop: 4 }}>{subtitle}</p>}
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
    </div>
  );
}
