import { NavLink, Route, Routes } from "react-router-dom";

import { AuthBadge } from "./components/AuthBadge";
import { PriceTickerStrip } from "./components/PriceTickerStrip";
import { AuthPage } from "./pages/AuthPage";
import { DashboardPage } from "./pages/DashboardPage";
import { SignalDetailPage } from "./pages/SignalDetailPage";

export default function App() {
  return (
    <div className="app-shell">
      <PriceTickerStrip />
      <header className="masthead">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true" />
          <div>
            <p className="eyebrow">Morocco Trading Agents</p>
            <h1>Operator Console</h1>
          </div>
        </div>
        <div className="masthead-actions">
          <nav className="top-nav" aria-label="Primary">
            <NavLink to="/" end>
              Dashboard
            </NavLink>
            <NavLink to="/auth">Access</NavLink>
          </nav>
          <AuthBadge />
        </div>
      </header>

      <main className="workspace">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="/signals/:requestId" element={<SignalDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}
