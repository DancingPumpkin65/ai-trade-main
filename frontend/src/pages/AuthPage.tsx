import { useState, useTransition } from "react";

import { useAuth } from "../lib/auth";
import { api } from "../lib/api";

export function AuthPage() {
  const { token, setToken, clearToken } = useAuth();
  const [username, setUsername] = useState("operator");
  const [password, setPassword] = useState("operator-pass");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function run(action: () => Promise<void>) {
    setError(null);
    setMessage(null);
    startTransition(() => {
      void action().catch((nextError) => {
        setError(nextError instanceof Error ? nextError.message : "Auth request failed.");
      });
    });
  }

  return (
    <div className="stack">
      <section className="panel panel-form">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Operator Access</p>
            <h2>Register or load a bearer token</h2>
          </div>
          <p className="support-copy">
            The current backend does not enforce auth on protected endpoints yet. This screen is ready for that switch and
            already stores the login token locally for future API enforcement.
          </p>
        </div>

        <form className="request-form">
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>

          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>

          <div className="form-footer field-span-2">
            <div className="action-row">
              <button
                className="ghost-button"
                disabled={isPending}
                onClick={(event) => {
                  event.preventDefault();
                  run(async () => {
                    const response = await api.register(username, password);
                    setMessage(`Registered ${response.username}.`);
                  });
                }}
                type="button"
              >
                Register
              </button>
              <button
                className="primary-button"
                disabled={isPending}
                onClick={(event) => {
                  event.preventDefault();
                  run(async () => {
                    const response = await api.login(username, password);
                    setToken(response.access_token);
                    setMessage("Token stored for frontend requests.");
                  });
                }}
                type="button"
              >
                {isPending ? "Working…" : "Login"}
              </button>
            </div>
            <button className="danger-button" onClick={clearToken} type="button">
              Clear token
            </button>
          </div>
        </form>

        {token ? <p className="token-block">{token}</p> : null}
        {message ? <p className="inline-success">{message}</p> : null}
        {error ? <p className="inline-error">{error}</p> : null}
      </section>
    </div>
  );
}
