import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/common/ProtectedRoute";
import AppLayout from "./components/layout/AppLayout";
import LoginPage from "./pages/LoginPage";
import DashboardOverviewPage from "./pages/DashboardOverviewPage";
import AlertsQueuePage from "./pages/AlertsQueuePage";
import AlertDetailPage from "./pages/AlertDetailPage";
import EntityHistoryPage from "./pages/EntityHistoryPage";
import NotFoundPage from "./pages/NotFoundPage";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />

          <Route
            element={
              <ProtectedRoute>
                <AppLayout />
              </ProtectedRoute>
            }
          >
            <Route path="/" element={<DashboardOverviewPage />} />
            <Route path="/alerts" element={<AlertsQueuePage />} />
            <Route path="/alerts/:logId" element={<AlertDetailPage />} />
            <Route path="/entities/:entityId" element={<EntityHistoryPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
