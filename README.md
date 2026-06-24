# TBCG — TechBro Consulting Group

**TBCG (TechBro Consulting Group)** is a multi-agent AI consulting firm. Submit a
business problem; a coordinated team of specialist agents researches, analyzes,
strategizes, assesses risk, critiques its own work, and (with your approval)
produces a consulting report.

Built with **LangGraph** orchestration, **HuggingFace Inference Providers** for
the LLM, a **fully-local RAG** stack (local embeddings + Dockerized ChromaDB),
and a **Streamlit** dashboard with live per-node streaming and PDF export.

The agents are **company-agnostic** — they consult on whatever company you
ingest into `knowledge_base/`. The default KB is a real small-cap public
company, **Purple Innovation (NASDAQ: PRPL)**, built from public filings so that
research/finance can corroborate real markets and competitors instead of
hallucinating. Swap in any company's docs and re-run `scripts/ingest_kb.py`.

## Quickstart (minimal)

Prerequisites: Python 3.10+, Docker, a HuggingFace token with Inference
Providers access.

```bash
python -m venv .venv && . .venv/Scripts/activate   # Unix: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                                # then set HF_TOKEN in .env
docker compose up -d && python scripts/ingest_kb.py # start Chroma + seed the KB

streamlit run app/streamlit_app.py                  # open http://localhost:8501
```

That's the whole thing. CLI alternative: `python -m tbcg.run "<your question>"`.

## Architecture

LangGraph state machine. The Engagement Manager classifies the problem and picks
the team; analysis agents fan out and converge on Strategy; Risk runs when
material; the Critic gates quality; high-impact calls pause for human approval.

```
Engagement Manager
   ├─ classify problem_type + pick team (+ conduct & risk screens)
   ▼
Research  ┐
Finance   ┘─(parallel, as required)─▶ Strategy ─▶ Risk* ─▶ Critic
                                                            │
                            approve ─▶ Human Review ─▶ Report ─▶ END
                            reject  ─▶ Revise ─▶ (dispatcher re-enters at the earliest
                                       stage the feedback affects; capped by MAX_REVISIONS)
   * Risk runs when the problem type requires it or the request is risk-material.
```

On a revision (critic or human feedback), a model-driven dispatcher reads the
feedback and re-enters the pipeline at the earliest stage that must change —
e.g. "expand on the strategy" re-runs Strategy only, preserving the research and
finance outputs — rather than blindly re-running everything.

| Agent | Role | Tools |
| --- | --- | --- |
| Engagement Manager | Classify problem, pick team, run conduct + risk-materiality screens | — (fast model) |
| Research | External (web) + internal (RAG) evidence, with provenance discipline | Web search (DuckDuckGo), RAG (Chroma) |
| Finance (+ operations) | ROI, break-even, revenue, cost levers; sources missing figures | Python REPL, RAG (baseline anchors), web search |
| Strategy | Synthesize the recommendation (no fabricated numbers) | — |
| Risk | Stress-test the strategy; lead with sanctions/conduct exposure when flagged | — |
| Critic | Independent quality gate (approve / revise); scope-, sanity-, provenance-aware | — |

**Problem types** (each maps to a deterministic minimum team in `policy.py`):
`market_entry`, `product_launch`, `cost_reduction`, `churn`, `market_compare`,
`pricing`, `investment`, `general`.

## Setup

### 1. Prerequisites
- Python 3.10+
- Docker (for ChromaDB)
- A HuggingFace token with Inference Providers access

### 2. Install
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env       # then edit .env and set HF_TOKEN
```
The app boots with only `HF_TOKEN` set. Defaults:
`HF_MODEL=deepseek-ai/DeepSeek-V3-0324` (reasoning) and
`HF_MODEL_FAST=meta-llama/Llama-3.3-70B-Instruct` (cheap classification).
Embeddings run locally (`all-MiniLM-L6-v2`) and cost no credits.

**Multiple tokens / rotation:** any env var whose value is an `hf_...` token is
auto-detected (regardless of name), and calls round-robin across them with
automatic failover on rate-limit/credit errors — just paste extra tokens into
`.env`.

### 4. Start the vector store + seed the knowledge base
```bash
docker compose up -d
python scripts/ingest_kb.py
```
RAG is optional — if Chroma is offline, the Research agent falls back to web
search only.

## Usage

### Streamlit dashboard (full experience: streaming + human approval + PDF)
```bash
streamlit run app/streamlit_app.py
```

### Headless CLI
```bash
python -m tbcg.run "Should we expand into the European mattress market?"
```

## Testing

Unit tests (no LLM/network, no credits — 48 tests):
```bash
pip install pytest
pytest                      # config + paths come from pyproject.toml
```

Live evaluation of the 5 canonical cases (spends credits, needs `HF_TOKEN`):
```bash
python tests/eval_cases.py
```

Lint / format (config in `pyproject.toml`):
```bash
ruff check . && black .
```

## Project layout
```
src/tbcg/
  config.py    env + settings (+ token rotation)   llm.py     HF LLM + structured output
  state.py     LangGraph state                      schemas.py Pydantic agent contracts
  policy.py    problem→team policy, sanctions/      run.py     headless CLI entry point
               conduct guardrails
  agents/      the six agents                       tools/     web_search, rag, python_repl, report
  rag/         embeddings + ingest                  graph/     build.py (wiring) + routing.py
app/streamlit_app.py   scripts/ingest_kb.py   knowledge_base/   tests/   pyproject.toml
```

## Cost notes
The only spend is HuggingFace LLM tokens (~$10–12 budget). Mitigations baked in:
local embeddings, a smaller model for classification, and a hard revision-loop
cap (`MAX_REVISIONS`) so the critic↔strategy loop can't run away.

## Observability (optional)
Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in
`.env` to trace agent runs, routing decisions, and tool calls in LangSmith. For
a non-US LangSmith org, also set the regional `LANGSMITH_ENDPOINT` (e.g.
`https://apac.api.smith.langchain.com`). Config mirrors these into the legacy
`LANGCHAIN_*` env vars so tracing works across langchain versions.

## Guardrails (defense in depth)
- **Schema bounds** reject impossible finance outputs (negative break-even, absurd ROI) → auto-retry.
- **RAG-grounded finance + web search** so figures come from real data, not fabrication; unsourced inputs are flagged for due diligence.
- **Problem policy** (`policy.py`) — deterministic minimum agent team per problem type.
- **Sanctioned-jurisdiction guardrail** — research/risk/finance lead with sanctions/legal exposure for embargoed markets.
- **Conduct/ethics guardrail** — illegal/deceptive/consumer-harmful requests force risk engagement and steer toward lawful alternatives.
- **Investment confidence cap** — high-stakes, data-poor decisions are capped pending due diligence.
- **Scope-aware critic** — only demands work the engaged team can produce; checks numeric sanity and provenance.
