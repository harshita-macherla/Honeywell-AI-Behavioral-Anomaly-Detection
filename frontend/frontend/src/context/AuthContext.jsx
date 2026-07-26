import { createContext, useContext, useEffect, useState } from "react";

/**
 * AuthContext
 * ===========
 * Demo authentication only, as explicitly scoped by this milestone -- the
 * FastAPI backend has no auth layer to integrate against yet, so this
 * intentionally does not call any API. It exists to gate the dashboard
 * behind a login screen and to give the UI an analyst identity to display
 * (top bar, entity-history "assigned analyst" style touches), matching
 * what a real enterprise console would show.
 *
 * Any non-empty username/password combination succeeds. Session persists
 * in localStorage (this is a standalone app, not an in-chat artifact, so
 * localStorage is appropriate here).
 */

const STORAGE_KEY = "sentinel.session";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [analyst, setAnalyst] = useState(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    if (analyst) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(analyst));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, [analyst]);

  const login = (username) => {
    setAnalyst({
      username,
      displayName: username
        .split(/[.\s_-]+/)
        .filter(Boolean)
        .map((part) => part[0].toUpperCase() + part.slice(1))
        .join(" "),
      loggedInAt: new Date().toISOString(),
    });
  };

  const logout = () => setAnalyst(null);

  return (
    <AuthContext.Provider value={{ analyst, isAuthenticated: !!analyst, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
