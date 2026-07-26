import { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { Radar, Lock, User } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { isAuthenticated, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  if (isAuthenticated) {
    const dest = location.state?.from?.pathname || "/";
    return <Navigate to={dest} replace />;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Enter an analyst ID and password to continue.");
      return;
    }
    login(username.trim());
    navigate(location.state?.from?.pathname || "/", { replace: true });
  }

  return (
    <div className="login-page">
      <div className="login-page__scene" aria-hidden="true">
        <div className="login-page__sweep" />
        <div className="login-page__grid" />
      </div>

      <div className="login-card panel">
        <div className="login-card__brand">
          <Radar size={26} color="var(--risk-critical)" />
          <div>
            <div style={{ fontWeight: 700, fontSize: "var(--fs-lg)" }}>Sentinel</div>
            <div className="eyebrow">Behavioral Anomaly Detection Console</div>
          </div>
        </div>

        <p style={{ color: "var(--text-secondary)", fontSize: "var(--fs-sm)", marginBottom: "var(--sp-5)" }}>
          Sign in to review live alerts across users, service accounts, and
          OT/IoT devices.
        </p>

        <form onSubmit={handleSubmit} noValidate>
          <div style={{ marginBottom: "var(--sp-4)" }}>
            <label className="field-label" htmlFor="username">
              Analyst ID
            </label>
            <div className="login-card__input-wrap">
              <User size={15} color="var(--text-tertiary)" />
              <input
                id="username"
                className="input login-card__input"
                placeholder="e.g. j.rivera"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          </div>

          <div style={{ marginBottom: "var(--sp-5)" }}>
            <label className="field-label" htmlFor="password">
              Password
            </label>
            <div className="login-card__input-wrap">
              <Lock size={15} color="var(--text-tertiary)" />
              <input
                id="password"
                type="password"
                className="input login-card__input"
                placeholder="\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <p role="alert" style={{ color: "var(--risk-high)", fontSize: "var(--fs-sm)", marginBottom: "var(--sp-4)" }}>
              {error}
            </p>
          )}

          <button type="submit" className="btn btn--primary btn--full">
            Sign in
          </button>
        </form>

        <p className="eyebrow" style={{ marginTop: "var(--sp-5)", textAlign: "center" }}>
          Demo authentication &middot; any credentials are accepted
        </p>
      </div>
    </div>
  );
}
