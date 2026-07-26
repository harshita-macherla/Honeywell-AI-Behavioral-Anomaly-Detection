import { AlertTriangle, Inbox, RefreshCw } from "lucide-react";

export function LoadingBlock({ label = "Loading\u2026" }) {
  return (
    <div className="state-block" role="status" aria-live="polite">
      <div className="skeleton" style={{ width: 28, height: 28, borderRadius: "50%" }} />
      <span className="eyebrow">{label}</span>
    </div>
  );
}

export function ErrorBlock({ error, onRetry }) {
  return (
    <div className="state-block state-block--error" role="alert">
      <AlertTriangle size={28} className="state-block__icon" />
      <div>
        <p style={{ fontWeight: 600, marginBottom: 4 }}>Couldn&rsquo;t load this data</p>
        <p style={{ fontSize: "var(--fs-sm)", color: "var(--text-secondary)" }}>
          {error?.message || "An unexpected error occurred."}
        </p>
      </div>
      {onRetry && (
        <button className="btn btn--sm" onClick={onRetry}>
          <RefreshCw size={14} /> Retry
        </button>
      )}
    </div>
  );
}

export function EmptyBlock({ title = "Nothing here", subtitle }) {
  return (
    <div className="state-block">
      <Inbox size={28} className="state-block__icon" />
      <div>
        <p style={{ fontWeight: 600, marginBottom: 4 }}>{title}</p>
        {subtitle && <p style={{ fontSize: "var(--fs-sm)", color: "var(--text-secondary)" }}>{subtitle}</p>}
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 6, cols = 6 }) {
  return (
    <div style={{ padding: "var(--sp-4)" }}>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} style={{ display: "flex", gap: "var(--sp-4)", marginBottom: "var(--sp-3)" }}>
          {Array.from({ length: cols }).map((__, c) => (
            <div key={c} className="skeleton" style={{ height: 16, flex: c === 0 ? "0 0 90px" : 1 }} />
          ))}
        </div>
      ))}
    </div>
  );
}
