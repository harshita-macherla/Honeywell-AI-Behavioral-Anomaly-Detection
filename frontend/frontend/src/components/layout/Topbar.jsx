import { useNavigate } from "react-router-dom";
import { useState } from "react";
import { Menu, Search, LogOut, CircleCheck, CircleAlert, CircleDashed } from "lucide-react";
import { useAuth } from "../../context/AuthContext";
import { useApi } from "../../hooks/useApi";
import { getStatus } from "../../api/client";

export default function Topbar({ onToggleSidebar }) {
  const navigate = useNavigate();
  const { analyst, logout } = useAuth();
  const [query, setQuery] = useState("");
  const { data: status, loading, error } = useApi(getStatus, []);

  function handleSearch(e) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    // Entity IDs in this dataset look like U0197 / IOT025 / EDG010 etc.
    // Alert log_ids are UUIDs. Route by shape, matching the backend's own
    // two lookup endpoints (entity history vs. alert detail).
    const looksLikeUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(trimmed);
    if (looksLikeUuid) {
      navigate(`/alerts/${trimmed}`);
    } else {
      navigate(`/entities/${trimmed}`);
    }
  }

  return (
    <header className="topbar">
      <button className="btn btn--ghost topbar__menu-btn" onClick={onToggleSidebar} aria-label="Toggle navigation">
        <Menu size={18} />
      </button>

      <form className="topbar__search" onSubmit={handleSearch} role="search">
        <Search size={15} color="var(--text-tertiary)" />
        <input
          className="topbar__search-input"
          placeholder="Search entity ID (U0197) or alert ID\u2026"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search entity or alert"
        />
      </form>

      <div className="topbar__right">
        <div className="topbar__status" title={error ? error.message : "Backend connection status"}>
          {loading ? (
            <CircleDashed size={14} color="var(--text-tertiary)" />
          ) : error ? (
            <CircleAlert size={14} color="var(--risk-critical)" />
          ) : (
            <CircleCheck size={14} color="var(--risk-low)" />
          )}
          <span className="mono">
            {loading ? "connecting\u2026" : error ? "API offline" : `API online \u00b7 ${status?.dataset_rows_loaded?.toLocaleString()} events`}
          </span>
        </div>

        <div className="topbar__user">
          <div className="topbar__avatar">{analyst?.displayName?.[0] || "?"}</div>
          <span className="topbar__username">{analyst?.displayName}</span>
          <button className="btn btn--ghost btn--sm" onClick={logout} aria-label="Log out">
            <LogOut size={14} />
          </button>
        </div>
      </div>
    </header>
  );
}
