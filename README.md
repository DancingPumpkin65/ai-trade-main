# Morocco Trading Agents

Simple trading-analysis platform for Moroccan stocks.

It has:
- a Python backend with FastAPI
- a React frontend with Bun + Vite
- SQLite by default
- optional PostgreSQL support
- optional Ollama-powered agent outputs
- Alpaca order preview, approval, and optional submission flow

The system can:
- analyze one symbol like `ATW`
- scan the Moroccan market for top opportunities
- understand prompt-style requests like `Analyze ATW with conservative risk`
- prepare Alpaca order commands after analysis

## Project Structure

- `trading_agents/`: backend code
- `frontend/`: web interface
- `tests/`: backend tests
- `harness/`: scenario-based evaluation runs
- `data/`: local runtime data

## Backend Setup

1. Create a virtual environment.

```bash
python -m venv .venv
```

2. Activate it.

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

3. Install backend dependencies.

```bash
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[dev,integrations]
pip install -e .[dev,postgres]
```

4. Copy the env file.

```bash
copy .env.example .env
```

5. Start the backend.

```bash
uvicorn trading_agents.api.main:app --reload
```

Backend URL:
- `http://localhost:8000`

API docs:
- `http://localhost:8000/docs`

## Frontend Setup

1. Open the frontend folder.

```bash
cd frontend
```

2. Install frontend dependencies.

```bash
bun install
```

3. Start the frontend.

```bash
bun run dev
```

Frontend URL:
- usually `http://localhost:5173`

## Environment

Important backend env vars:

- `DB_PATH`: SQLite file path
- `DATABASE_URL`: optional PostgreSQL connection string
- `DRAHMI_API_KEY`: Drahmi API key
- `MARKETAUX_API_KEY`: MarketAux API key
- `OLLAMA_BASE_URL`: Ollama server URL
- `OLLAMA_MODEL`: Ollama model name
- `AGENT_LLM_ENABLED=true|false`: enable LLM-powered agent bodies
- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- `ALPACA_BASE_URL`
- `ALPACA_REQUIRE_ORDER_APPROVAL=true|false`
- `ALPACA_SUBMIT_ORDERS=true|false`

If no Drahmi key is configured, the app uses sample market data so it still runs locally.

## Database

Run migrations:

```bash
python migrate_db.py
```

Or if installed as a package:

```bash
morocco-trading-migrate
```

Default database:
- SQLite

Production option:
- PostgreSQL with `DATABASE_URL=postgresql://...`

## Run Tests

Run all backend tests:

```bash
python -m pytest
```

Run frontend tests:

```bash
cd frontend
bun run test
```

Run frontend lint:

```bash
cd frontend
bun run lint
```

Build frontend:

```bash
cd frontend
bun run build
```

## Harness

Run the scenario harness:

```bash
python -m harness.run_harness --scenario all
```

Replay a saved report:

```bash
python -m harness.run_harness --replay harness/reports/smoke-report.json
```

## Typical Flow

1. User sends a request
2. Backend parses the intent
3. Market/news/documents are loaded
4. Sentiment, technical, risk, and coordinator agents run
5. Python guardrails enforce limits
6. Final signal is saved
7. Alpaca order preview is prepared
8. Operator approves or rejects the order
9. Optional paper/live Alpaca submission runs if enabled

## Current Notes

- Analysis runs automatically to completion.
- Approval happens on the Alpaca order command, not in the middle of analysis.
- LLM-powered agent bodies are optional and fall back to deterministic logic if disabled or unavailable.
- The app is usable locally with SQLite and sample data.
