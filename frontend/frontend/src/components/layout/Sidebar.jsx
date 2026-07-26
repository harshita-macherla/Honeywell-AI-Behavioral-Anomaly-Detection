import { NavLink } from "react-router-dom";
import { LayoutDashboard, ListFilter, ShieldAlert, Radar } from "lucide-react";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/alerts", label: "Alert Queue", icon: ListFilter },
];

export default function Sidebar({ open, onNavigate }) {
  return (
    <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
      <div className="sidebar__brand">
        <Radar size={22} color="var(--risk-critical)" />
        <div>
          <div style={{ fontWeight: 700, fontSize: "var(--fs-md)", letterSpacing: "-0.01em" }}>Sentinel</div>
          <div className="eyebrow">Behavioral Anomaly Detection</div>
        </div>
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) => `sidebar__link ${isActive ? "sidebar__link--active" : ""}`}
          >
            <Icon size={17} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__footer">
        <div className="sidebar__footer-row">
          <ShieldAlert size={14} color="var(--risk-low)" />
          <span>v2 pipeline &middot; live</span>
        </div>
      </div>
    </aside>
  );
}
