import { useState, useTransition } from "react";

import type { GenerateSignalRequest, RiskPreference, TimeHorizon } from "../lib/types";

interface RequestComposerProps {
  onSubmit: (payload: GenerateSignalRequest) => Promise<void>;
  onSubmitLive: (payload: GenerateSignalRequest) => Promise<void>;
}

const riskOptions: RiskPreference[] = ["CONSERVATIVE", "BALANCED", "AGGRESSIVE"];
const horizonOptions: TimeHorizon[] = ["UNSPECIFIED", "INTRADAY", "SHORT_TERM", "SWING"];

export function RequestComposer({ onSubmit, onSubmitLive }: RequestComposerProps) {
  const [symbol, setSymbol] = useState("");
  const [capital, setCapital] = useState("100000");
  const [prompt, setPrompt] = useState("Analyze ATW with conservative risk");
  const [riskProfile, setRiskProfile] = useState<RiskPreference | "">("");
  const [timeHorizon, setTimeHorizon] = useState<TimeHorizon | "">("");
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  function buildPayload(): GenerateSignalRequest {
    setError(null);

    const payload: GenerateSignalRequest = {};
    if (symbol.trim()) {
      payload.symbol = symbol.trim().toUpperCase();
    }
    if (capital.trim()) {
      payload.capital = Number(capital);
    }
    if (prompt.trim()) {
      payload.prompt = prompt.trim();
    }
    if (riskProfile) {
      payload.risk_profile = riskProfile;
    }
    if (timeHorizon) {
      payload.time_horizon = timeHorizon;
    }
    return payload;
  }

  function runSubmission(handler: (payload: GenerateSignalRequest) => Promise<void>) {
    const payload = buildPayload();

    startTransition(() => {
      void handler(payload).catch((submitError) => {
        setError(submitError instanceof Error ? submitError.message : "Unable to launch analysis.");
      });
    });
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    runSubmission(onSubmit);
  }

  return (
    <section className="panel panel-form">
      <div className="section-heading">
        <div>
          <p className="eyebrow">New Analysis</p>
          <h2>Compose an operator request</h2>
        </div>
        <p className="support-copy">
          Structured inputs and prompt-first instructions can be mixed in the same request.
        </p>
      </div>

      <form className="request-form" onSubmit={handleSubmit}>
        <label>
          Symbol
          <input
            name="symbol"
            placeholder="ATW"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
          />
        </label>

        <label>
          Capital (MAD)
          <input
            inputMode="numeric"
            name="capital"
            placeholder="100000"
            value={capital}
            onChange={(event) => setCapital(event.target.value)}
          />
        </label>

        <label className="field-span-2">
          Prompt
          <textarea
            name="prompt"
            rows={4}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>

        <label>
          Risk profile
          <select value={riskProfile} onChange={(event) => setRiskProfile(event.target.value as RiskPreference | "")}>
            <option value="">Use parser default</option>
            {riskOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label>
          Time horizon
          <select value={timeHorizon} onChange={(event) => setTimeHorizon(event.target.value as TimeHorizon | "")}>
            <option value="">Use parser default</option>
            {horizonOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <div className="form-footer field-span-2">
          <p className="micro-copy">
            Example: “I have 100,000 MAD. What are the best possible trades this week?”
          </p>
          <div className="action-row">
            <button className="ghost-button" disabled={isPending} type="submit">
              Snapshot run
            </button>
            <button
              className="primary-button pill-button"
              disabled={isPending}
              onClick={(event) => {
                event.preventDefault();
                runSubmission(onSubmitLive);
              }}
              type="button"
            >
              {isPending ? "Launching…" : "Run live analysis"}
            </button>
          </div>
        </div>
      </form>

      {error ? <p className="inline-error">{error}</p> : null}
    </section>
  );
}
