# Root AGENTS.md for TradingAgents

Date: 2026-09-01
Status: approved design, pending implementation
Audience: coding agents (Grok, Cursor, Codex, Claude Code)

## Goal

Add a single root `AGENTS.md` that coding agents load automatically. The file
teaches them how to edit this repository: layout, commands, style, graph
ownership, and the invariants that keep getting broken. It is project rules,
not a README.

## Non-goals

- Human-facing product documentation (that stays in `README.md` / `CHANGELOG.md`).
- Nested `AGENTS.md` files under `tradingagents/`, `cli/`, or `tests/`.
- A Grok agent definition under `.grok/agents/`.
- Copying API-key setup, Docker walkthroughs, model catalogs, or the paper.
- Live brokerage, order routing, or any change to runtime trading behavior.

## File identity

- **Path:** `AGENTS.md` at the repository root.
- **Size:** about 100–150 lines. Grok loads the whole file every session.
- **Tone:** imperative rules. Prefer "do X; the owner is `path`" over essays.
- **Maintenance:** when an invariant in this file is no longer true, update
  `AGENTS.md` in the same PR as the code. The file is current rules, not a
  snapshot of v0.4.0 trivia.

## Document outline

Eight sections, in this order:

1. What this repo is
2. Layout
3. Commands
4. Style
5. Architecture
6. Invariants
7. Testing
8. Out of scope

## Section content

### 1. What this repo is

Two sentences: TradingAgents is a LangGraph multi-agent research framework
that runs analyst → research debate → trader → risk debate → portfolio
manager. It is a research tool, not trading advice.

Do not restate the analyst roster, Discord links, or citation block.

### 2. Layout

A compact map, not a tree dump:

- `tradingagents/graph/` — graph wiring, checkpoints, routers, reflection,
  `TradingAgentsGraph`
- `tradingagents/agents/` — agent factories, Pydantic schemas, tools, memory
  log, rating vocabulary
- `tradingagents/dataflows/` — vendor implementations, `interface.py` routing,
  look-ahead date window
- `tradingagents/llm_clients/` — lazy provider factory, capabilities, model
  catalog
- `cli/` — Typer CLI; must share checkpoint lifecycle with `propagate()`
- `tests/` — pytest; dummy API keys and isolated config are autouse fixtures
- `tradingagents/default_config.py` — config source of truth

### 3. Commands

Match CI. Do not invent wrappers.

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
pip install .
python -c "import tradingagents, cli.main"
```

Notes to include:

- CI runs pytest on Python 3.10–3.13 and a clean-install import smoke.
- `pip install -e ".[bedrock]"` only when touching Bedrock.
- Docker and the `tradingagents` CLI live in `README.md`; link, do not repeat.

### 4. Style

- Python 3.10+, line length 100.
- Ruff select `E,W,F,I,B,UP,C4,SIM`; `E501` ignored. Source of truth:
  `pyproject.toml`.
- Do not run repo-wide `ruff format` (explicitly deferred to avoid mass diffs).
- `isort` combine-as-imports: keep aliased re-exports in one block
  (`dataflows/interface.py` is the example).
- `__init__.py` re-exports are intentional (`F401` ignored per file).
- Prefer extending an existing helper over a parallel one:
  `date_window`, `structured.py`, `rating.py`, `safe_ticker_component`.

### 5. Architecture

Runtime flow (one LangGraph):

1. Analysts in sequence: market → sentiment (`social` wire name) → news →
   fundamentals. Each analyst is three nodes: agent → tools → message-clear.
2. Research debate: Bull ↔ Bear, then Research Manager (structured investment
   plan, 5-tier rating).
3. Trader (structured Buy/Hold/Sell). Must receive the technical market
   report, not only the digested plan, so prices/stops are grounded.
4. Risk debate: Aggressive / Conservative / Neutral, then Portfolio Manager
   (structured 5-tier decision).
5. Signal processor reads the PM markdown with `extract_rating`. Unparseable
   → `REVIEW`, never a silent `Hold`.

Config:

- `tradingagents/default_config.py` is the source of truth.
- New env-overridable keys go in `_ENV_OVERRIDES` there. Do not special-case
  them in CLI scripts.
- `dataflows.config.set_config` merges; tests must replace the global, not
  only overlay keys.

### 6. Invariants

Each bullet in `AGENTS.md` is a rule plus the owning module.

1. **No look-ahead.** Dated news/social content uses
   `tradingagents/dataflows/date_window.py` (UTC, half-open
   `[start, end+1 day)`). FRED pins vintage to the as-of date. Alpha Vantage
   fundamentals must be parsed then filtered. Memory `get_past_context` only
   returns lessons resolved by the trade date. Do not settle a decision before
   the holding window has fully traded.
2. **Structured decision agents.** Research Manager, Trader, and Portfolio
   Manager go through `tradingagents/agents/schemas.py` and
   `tradingagents/agents/utils/structured.py`
   (`bind_structured` / `invoke_structured_or_freetext`). Render back to
   markdown. On structured failure, fall back to free text; do not block the
   graph. Those agents get `NO_EXTERNAL_TOOLS`; they do not grow search tools.
3. **One rating vocabulary.** 5-tier lives in
   `tradingagents/agents/utils/rating.py` (`Buy`, `Overweight`, `Hold`,
   `Underweight`, `Sell`). Trader is 3-tier (`Buy`/`Hold`/`Sell`). New call
   sites use `extract_rating` (None → `REVIEW`). `parse_rating` is legacy
   silent-default only.
4. **Complete router maps.** `DEBATE_PATH_MAP` and `RISK_ANALYSIS_PATH_MAP`
   in `tradingagents/graph/setup.py` must list every target the shared routers
   can return. The first speaker in a debate opens with their own case; they
   do not rebut an empty opponent.
5. **Vendor routing is centralized.** New data tools/vendors register in
   `tradingagents/dataflows/interface.py` (`VENDOR_METHODS`, categories). Core
   categories fail loud; `macro_data` and `prediction_markets` are optional
   and degrade. Agents must not call a vendor module, bypassing the interface.
6. **LLM factory stays lazy.** `tradingagents/llm_clients/factory.py` imports
   provider modules inside the function. New providers go through the
   factory/registry. OpenAI-compatible endpoints use that path, not a one-off
   client.
7. **Identity and paths.** Resolve instrument identity before agents run.
   Ticker strings that hit the filesystem go through `safe_ticker_component`.
8. **Checkpoints.** CLI `--checkpoint` and `propagate()` share one lifecycle.
   A resume run must continue the interrupted graph, not duplicate messages.

### 7. Testing

- New behavior gets a test under `tests/`, named after the bug or feature,
  following nearby files.
- Unit tests must not need live API keys or network. `tests/conftest.py`
  already injects placeholder keys and isolates `dataflows` config.
- Use the `mock_llm_client` fixture instead of constructing providers.
- Mark with `unit` / `integration` / `smoke` when it helps; default is fast
  unit.
- Look-ahead, rating, structured-output, and router-map changes need a
  regression test that would have failed before the fix.
- Do not add tests that call real LLM or market APIs.

### 8. Out of scope

- Do not duplicate README, CHANGELOG, or the paper.
- Do not add live brokerage, order routing, or extra "this is trading advice"
  copy beyond the one-line research-tool disclaimer.
- Do not call vendor modules from agents; go through `interface.py`.
- Do not add a new LLM provider without the factory/registry and a test.
- Do not special-case new config keys in the CLI; add them to `_ENV_OVERRIDES`.
- Do not nest more `AGENTS.md` files in this pass.
- Do not commit API keys, `results/`, or checkpoint DBs.

## Implementation shape

This is a documentation-only change:

- Create `AGENTS.md` at the repo root.
- Do not change Python packages, CI, `pyproject.toml`, or `README.md`.
- No automated test for the markdown file. Verification is: file exists at
  root, covers the eight sections, stays within ~150 lines, and does not
  contradict `pyproject.toml` or CI commands.

## Success criteria

- Coding agents in this repo can find the graph, vendors, and config without
  rereading the README.
- The eight invariants above are stated as rules with owning modules.
- Commands in the file match `.github/workflows/ci.yml` and `pyproject.toml`.
- The file does not grow into a second README.
