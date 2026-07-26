/**
 * client.js
 * =========
 * Thin fetch wrapper around every endpoint exposed by the FastAPI backend
 * (backend/app/main.py). This is the ONLY module in the frontend that
 * knows about HTTP -- pages and components call these functions and get
 * back plain JS objects (or throw an ApiError), never touching fetch/URL
 * construction directly. No backend or ML code is modified or duplicated
 * here; this purely consumes the existing REST surface documented in the
 * Backend API milestone.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkErr) {
    throw new ApiError(
      `Could not reach the API at ${BASE_URL}. Is the backend running?`,
      0,
      networkErr.message
    );
  }

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const body = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail
        ? JSON.stringify(detail)
        : `Request to ${path} failed (${response.status})`;
    throw new ApiError(message, response.status, detail);
  }

  return body;
}

function toQueryString(params = {}) {
  const usable = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  if (usable.length === 0) return "";
  return "?" + new URLSearchParams(usable).toString();
}

// ---------------------------------------------------------------------------
// Health / status
// ---------------------------------------------------------------------------
export const getHealth = () => request("/health");
export const getStatus = () => request("/api/v1/status");

// ---------------------------------------------------------------------------
// Prediction (live scoring)
// ---------------------------------------------------------------------------
export const predictEvent = (payload) =>
  request("/api/v1/predict", { method: "POST", body: JSON.stringify(payload) });

// ---------------------------------------------------------------------------
// Explanation
// ---------------------------------------------------------------------------
export const explainLogId = (logId) => request(`/api/v1/explain/${encodeURIComponent(logId)}`);

// ---------------------------------------------------------------------------
// Analyst dashboard endpoints
// ---------------------------------------------------------------------------
export const listAlerts = (params = {}) => request(`/api/v1/alerts${toQueryString(params)}`);

export const getAlert = (logId) => request(`/api/v1/alerts/${encodeURIComponent(logId)}`);

export const getEntityHistory = (entityId, params = {}) =>
  request(`/api/v1/entities/${encodeURIComponent(entityId)}/history${toQueryString(params)}`);

export const getStatsOverview = () => request("/api/v1/stats/overview");
