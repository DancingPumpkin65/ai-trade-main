import { useAuth } from "../lib/auth";

export function AuthBadge() {
  const { token, clearToken } = useAuth();

  if (!token) {
    return <span className="status-badge tone-neutral">Guest mode</span>;
  }

  return (
    <div className="auth-badge">
      <span className="status-badge tone-success">Token loaded</span>
      <button className="ghost-button" onClick={clearToken} type="button">
        Clear
      </button>
    </div>
  );
}
