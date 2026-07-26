import { useCallback, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { listAlerts } from "../api/client";
import RiskBadge from "../components/common/RiskBadge";
import { EmptyBlock, ErrorBlock, TableSkeleton } from "../components/common/StateBlocks";
import { PageHeader } from "./DashboardOverviewPage";
import { ENTITY_TYPES, RISK_LEVELS, entityTypeLabel, formatTimestamp, truncateId } from "../utils/format";

const PAGE_SIZE = 25;

export default function AlertsQueuePage() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [entitySearch, setEntitySearch] = useState(params.get("entity_id") || "");

  const riskLevel = params.get("risk_level") || "";
  const entityType = params.get("entity_type") || "";
  const entityIdFilter = params.get("entity_id") || "";
  const page = parseInt(params.get("page") || "0", 10);

  function updateParam(key, value) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    next.delete("page");
    setParams(next);
  }

  function setPage(newPage) {
    const next = new URLSearchParams(params);
    next.set("page", String(newPage));
    setParams(next);
  }

  function handleSearchSubmit(e) {
    e.preventDefault();
    updateParam("entity_id", entitySearch.trim());
  }

  function clearFilters() {
    setEntitySearch("");
    setParams({});
  }

  const fetcher = useCallback(
    () =>
      listAlerts({
        risk_level: riskLevel || undefined,
        entity_type: entityType || undefined,
        entity_id: entityIdFilter || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    [riskLevel, entityType, entityIdFilter, page]
  );
  const { data, loading, error, reload } = useApi(fetcher, [riskLevel, entityType, entityIdFilter, page]);

  const totalPages = useMemo(() => (data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1), [data]);
  const hasActiveFilters = riskLevel || entityType || entityIdFilter;

  return (
    <div className="page">
      <PageHeader title="Alert Queue" subtitle="Ranked by risk score across every monitored entity" />

      <section className="panel panel--padded">
        <div className="filter-bar">
          <form onSubmit={handleSearchSubmit} className="topbar__search" style={{ maxWidth: 260, height: 34 }}>
            <Search size={14} color="var(--text-tertiary)" />
            <input
              className="topbar__search-input"
              placeholder="Filter by entity ID\u2026"
              value={entitySearch}
              onChange={(e) => setEntitySearch(e.target.value)}
            />
          </form>

          <div className="filter-bar__group">
            {RISK_LEVELS.map((level) => (
              <button
                key={level}
                className={`chip ${riskLevel === level ? "chip--active" : ""}`}
                onClick={() => updateParam("risk_level", riskLevel === level ? "" : level)}
              >
                {level}
              </button>
            ))}
          </div>

          <select
            className="select"
            value={entityType}
            onChange={(e) => updateParam("entity_type", e.target.value)}
            aria-label="Filter by entity type"
          >
            <option value="">All entity types</option>
            {ENTITY_TYPES.map((t) => (
              <option key={t} value={t}>
                {entityTypeLabel(t)}
              </option>
            ))}
          </select>

          {hasActiveFilters && (
            <button className="btn btn--ghost btn--sm" onClick={clearFilters}>
              <X size={13} /> Clear filters
            </button>
          )}
        </div>
      </section>

      <section className="panel">
        {loading ? (
          <TableSkeleton rows={10} cols={6} />
        ) : error ? (
          <ErrorBlock error={error} onRetry={reload} />
        ) : data.alerts.length === 0 ? (
          <EmptyBlock title="No alerts match these filters" subtitle="Try clearing a filter or searching a different entity ID." />
        ) : (
          <>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Risk</th>
                  <th>Entity</th>
                  <th>Type</th>
                  <th>Predicted attack</th>
                  <th>Confidence</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {data.alerts.map((alert) => (
                  <tr key={alert.log_id} onClick={() => navigate(`/alerts/${alert.log_id}`)}>
                    <td>
                      <RiskBadge level={alert.risk_level} />
                      <span className="mono" style={{ marginLeft: 8, fontSize: "var(--fs-xs)", color: "var(--text-secondary)" }}>
                        {alert.risk_score.toFixed(1)}
                      </span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontWeight: 600 }}>
                        {alert.entity_id}
                      </span>
                      {alert.department && (
                        <div style={{ fontSize: "var(--fs-xs)", color: "var(--text-tertiary)" }}>{alert.department}</div>
                      )}
                    </td>
                    <td>{entityTypeLabel(alert.entity_type)}</td>
                    <td>
                      {alert.predicted_attack_type ? alert.predicted_attack_type.replace(/_/g, " ") : (
                        <span style={{ color: "var(--text-tertiary)" }}>None</span>
                      )}
                    </td>
                    <td className="mono">{(alert.prediction_confidence * 100).toFixed(1)}%</td>
                    <td className="mono" style={{ color: "var(--text-secondary)" }}>
                      {formatTimestamp(alert.timestamp)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="pagination">
              <span>
                {data.total.toLocaleString()} alert{data.total === 1 ? "" : "s"} &middot; page {page + 1} of {totalPages}
              </span>
              <div className="pagination__controls">
                <button className="btn btn--sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
                  <ChevronLeft size={14} /> Prev
                </button>
                <button className="btn btn--sm" disabled={page + 1 >= totalPages} onClick={() => setPage(page + 1)}>
                  Next <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
