import { useCallback } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronRight, History, ShieldCheck, ShieldX } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { explainLogId } from "../api/client";
import RiskBadge from "../components/common/RiskBadge";
import { ErrorBlock, LoadingBlock } from "../components/common/StateBlocks";
import RiskGauge from "../components/viz/RiskGauge";
import ShapChart from "../components/viz/ShapChart";
import { entityTypeLabel, formatTimestamp } from "../utils/format";

export default function AlertDetailPage() {
  const { logId } = useParams();
  const fetcher = useCallback(() => explainLogId(logId), [logId]);
  const { data: alert, loading, error, reload } = useApi(fetcher, [logId]);

  if (loading) return <LoadingBlock label="Explaining alert\u2026" />;
  if (error) return <ErrorBlock error={error} onRetry={reload} />;
  if (!alert) return null;

  const wasCorrect =
    alert.actual_attack_type !== null && alert.predicted_attack_type === alert.actual_attack_type;

  return (
    <div className="page">
      <div className="breadcrumb">
        <Link to="/alerts">Alert Queue</Link>
        <ChevronRight size={14} />
        <span className="mono">{alert.log_id}</span>
      </div>

      <div className="page-header">
        <div>
          <h1 style={{ fontSize: "var(--fs-xl)", display: "flex", alignItems: "center", gap: 12 }}>
            {alert.predicted_attack_type ? alert.predicted_attack_type.replace(/_/g, " ") : "No attack predicted"}
            <RiskBadge level={alert.risk_level} />
          </h1>
          <p style={{ color: "var(--text-secondary)", marginTop: 4 }}>
            {entityTypeLabel(alert.entity_type)}{" "}
            <Link to={`/entities/${alert.entity_id}`} className="mono">
              {alert.entity_id}
            </Link>{" "}
            &middot; {formatTimestamp(alert.timestamp)}
          </p>
        </div>
        <div className="page-header__actions">
          <Link className="btn" to={`/entities/${alert.entity_id}`}>
            <History size={15} /> Entity history
          </Link>
        </div>
      </div>

      <div className="detail-grid">
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-5)" }}>
          <section className="panel panel--padded">
            <div className="panel__header">
              <h2 style={{ fontSize: "var(--fs-md)" }}>Why this was flagged</h2>
            </div>
            <ul className="reason-list" style={{ listStyle: "none", padding: 0, margin: "0 0 var(--sp-5) 0" }}>
              {alert.merged_reasons.map((reason) => (
                <li key={reason} className="reason-item">
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--risk-critical)", flexShrink: 0 }} />
                  {reason}
                </li>
              ))}
            </ul>

            <h3 style={{ fontSize: "var(--fs-sm)", color: "var(--text-secondary)", marginBottom: "var(--sp-3)" }}>
              SHAP feature attribution
            </h3>
            <ShapChart contributions={alert.top_shap_contributions} />
          </section>

          <section className="panel panel--padded">
            <div className="panel__header">
              <h2 style={{ fontSize: "var(--fs-md)" }}>Rule-engine flags</h2>
            </div>
            {alert.rule_reasons.length === 0 ? (
              <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)" }}>No explicit rules triggered.</p>
            ) : (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "var(--sp-2)" }}>
                {alert.rule_reasons.map((r) => (
                  <span key={r} className="chip chip--active" style={{ cursor: "default" }}>
                    {r}
                  </span>
                ))}
              </div>
            )}
          </section>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sp-5)" }}>
          <section className="panel panel--padded" style={{ display: "flex", justifyContent: "center" }}>
            <RiskGauge score={alert.risk_score} level={alert.risk_level} />
          </section>

          <section className="panel panel--padded">
            <h3 style={{ fontSize: "var(--fs-sm)", color: "var(--text-secondary)", marginBottom: "var(--sp-3)" }}>
              Classification
            </h3>
            <dl className="kv-grid">
              <div>
                <dt>Predicted attack type</dt>
                <dd>{alert.predicted_attack_type ? alert.predicted_attack_type.replace(/_/g, " ") : "None"}</dd>
              </div>
              <div>
                <dt>Model confidence</dt>
                <dd className="mono">{(alert.prediction_confidence * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Ground truth (report-only)</dt>
                <dd>{alert.actual_attack_type ? alert.actual_attack_type.replace(/_/g, " ") : "None"}</dd>
              </div>
              <div>
                <dt>Prediction outcome</dt>
                <dd style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  {alert.actual_attack_type === null ? (
                    "\u2014"
                  ) : wasCorrect ? (
                    <>
                      <ShieldCheck size={14} color="var(--risk-low)" /> Correct
                    </>
                  ) : (
                    <>
                      <ShieldX size={14} color="var(--risk-critical)" /> Missed
                    </>
                  )}
                </dd>
              </div>
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}
