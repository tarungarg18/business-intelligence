# Verity — KPI Intelligence-to-Action Engine

Accenture Innovation Challenge 2026 · Problem Track 3: **BusinessIntelligence.ai**

Verity turns a KPI movement into a **governed decision**: detect whether the
change is material, explain *why* with ranked drivers and cited evidence,
recommend *what to do* with the right approvals — or **abstain** when the
evidence is insufficient or contradictory.

## The central rule

> **The LLM is never the source of quantitative truth.**

Detection, forecasting, attribution, retrieval, ranking, scoring, permissions,
policy checks, and simulation are **deterministic**. The language model only
receives a finished **Evidence Pack** and phrases a narrative or boardroom memo.
Every draft is checked by a **citation guard**: invented evidence IDs or numbers
are rejected, and the system falls back to a deterministic template.

---

## What this repository does

Enterprises already have dashboards that show “revenue fell.” They still spend
hours answering the real questions:

1. Is this unusual, or just seasonality?
2. Does it matter commercially?
3. Which drivers contributed, and how much is unexplained interaction?
4. What documents and policies support that story?
5. Are we confident enough to act — or should we abstain?
6. What action, who owns it, who must approve it, what trade-off?

**Verity answers that sequence end to end** on a synthetic but realistic world
(ERP sales, marketing API, CRM churn, ops tickets, news, policies) so every
claim can be checked against **planted ground truth**.

### Pipeline (one request)

```text
Principal (who is asking)
        │
        ▼
Governed KPI read (DuckDB + RBAC) ── deny? → audited AccessDenied
        │
        ▼
Analytical engine
  · ETS baseline (fitted only on history *before* the window)
  · STL residual + Isolation Forest
  · Dual materiality (statistical + business impact)
        │
        ▼
Attribution (ranked drivers + unexplained residual)
        │
        ▼
Evidence Engine (RBAC filter → retrieve → 5-factor rerank → contradictions)
        │
        ▼
Evidence Pack  ──►  narrative (LLM + citation guard, or template)
                 └──► Cost Governor tier
                        · Tier 0  rules only (no LLM / no War Room)
                        · Tier 2  evidence narrative
                        · Tier 3  Decision War Room → action payload
```

### What is deterministic vs LLM

| Deterministic (truth) | LLM (phrasing only) |
|---|---|
| KPI values, materiality, ETS forecast | Investigation narrative wording |
| STL / Isolation Forest flags | War Room memo wording |
| Driver contributions + residual | Persona tone (CFO / analyst / ops) |
| Retrieval ranking & confidence | — |
| Policy authority & escalation | — |
| What-if simulation effects | — |
| RBAC allow / deny + audit | — |

---

## How it works for every demo case

Use these in the Streamlit UI (**Scenario** + **User**). Prefer **User = analyst**
unless you are demonstrating RBAC.

### S1 — West multi-factor shock → **explain and decide**

| | |
|---|---|
| **What happens** | West `net_revenue` falls ~12% in one week. |
| **Planted causes** | Inventory (dominant) → promotion lapse → competitor (weakest). |
| **System behaviour** | Detected + material + critical. Drivers ranked with an **interaction residual** (not forced to 100%). Evidence cites ops / tickets / promo / news. Cost Governor routes to **War Room**. Action e.g. expedite inventory under policy **P031**, with owner, approval, 24h monitoring. |
| **What to notice** | Residual row; evidence scores; action payload JSON; route = `decision_war_room`. |

### S2 — North contradiction → **abstain**

| | |
|---|---|
| **What happens** | North revenue dips, but documents disagree (tickets vs “no incident”, stale marketing note). |
| **Planted truth** | Cause is **not** recoverable from evidence on purpose. |
| **System behaviour** | Movement may still be detected, but **contradictions** fire, confidence drops, **`should_abstain`**. Narrative refuses a root-cause claim. War Room returns **Human review required** — no forced action. |
| **What to notice** | Yellow contradiction warning; no confident “the cause is X”. |

### S3 — New SKU (sparse history) → **explain, hedge**

| | |
|---|---|
| **What happens** | Newly launched SKU-E units spike under a launch promo; &lt; ~6 weeks of history. |
| **System behaviour** | Explains with promotion as driver, but **confidence is capped** (≤ 55%) and history is marked insufficient. Honest uncertainty, not overconfidence. |
| **What to notice** | Low confidence tile; thin-history behaviour. |

### S4 — East control → **stay silent**

| | |
|---|---|
| **What happens** | Ordinary East week; **no** planted shock. |
| **System behaviour** | Not material → route **`rules_only`**. No War Room, no LLM spend for noise. Used to measure **false-alarm rate**. |
| **What to notice** | Near-zero movement; no decision panel. |

### RBAC — West manager cannot read North / East

| | |
|---|---|
| **How to demo** | User = `west_manager`, Scenario = S2 (North) or S4 (East). |
| **System behaviour** | Governed query raises **AccessDenied** (row-level policy). UI shows a clear governance message instead of crashing. Same principal also filters documents **before** ranking — the AI cannot see what the human cannot see. |
| **Audit tab** | Button “Exercise West manager denied East read” writes a denial into the audit log. |

### Personas (same Evidence Pack)

| Persona | Intent |
|---|---|
| `analyst` | Precise drivers + citations |
| `cfo` | Short, decision-dense summary |
| `ops` | Action-oriented wording |

Truth does not change — only depth and tone.

---

## Quick start for (clone → run)

**Requirements:** Python **3.11+**, internet only for `pip install` (API keys optional).

All commands below are run from the **repository root** (the folder that contains
`verity/`, `requirements.txt`, and this `README.md`).

### 1. Clone and enter the repo

```bash
git clone https://github.com/tarungarg18/business-intelligence.git
cd "Business Intelligence"
```


### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. API keys for live LLM narratives

```bash
# Windows
copy .env.example .env
```

Edit `.env` 

```text
GEMINI_API_KEY=...       # primary embeddings + generation
OPENROUTER_API_KEY=...   # generation fallback
HF_TOKEN=...             # Hugging Face generation fallback
```

With **no** keys: offline LSA-style embeddings + deterministic template generator.

### 5. Build the synthetic warehouse (first time)

```bash
python -m verity.scripts build
```

This generates ERP / promotion / CRM data and ground truth into DuckDB
(`data/warehouse.duckdb`). Deterministic for seed `20260822`.

### 6. Run the demo UI 

```bash
streamlit run verity/app/ui.py
```

Open **http://localhost:8501** in a browser.

Suggested first path: **User = analyst**, **Scenario = S1**, then try S2 → S4 →
`west_manager` + S2 for RBAC.

### 7. Other ways to run

**CLI (prints assessment, narrative, decision):**

```bash
python -m verity.scripts demo
# or a specific scenario:
python -m verity.scripts demo --scenario S2
```

**REST API:**

```bash
uvicorn verity.app.api:app --reload
```

Then open **http://localhost:8000/docs** (Swagger) or call:

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness |
| `GET /scenarios` | list synthetic scenarios |
| `GET /scenario/S1?principal=analyst&persona=analyst` | full pipeline for S1 |
| `GET /lineage/net_revenue` | KPI lineage + definition conflict |
| `GET /evaluation` | measured metrics |
| `GET /model-health` | health / drift panel values |
| `GET /audit` | audit log |
| `POST /audit/exercise-denial` | force a West→East denial |
| `GET /risk-radar` | scenario risk strip |

**Tests (prove the prototype):**

```bash
python -m pytest
```

Expect on the order of **90+** passing tests (no network required).


## Repository layout

```text
verity/
  analytics/         STL, Isolation Forest, ETS, attribution, PVM, what-if
  app/               service.py · api.py (FastAPI) · ui.py (Streamlit)
  configs/           kpis.yaml (semantic contract) · policies.yaml (decision rights)
  datagen/           synthetic world, scenarios, documents, ground truth
  governance/        RBAC, audit, cost governor
  llm/               provider chain + offline floor
  rag/               Evidence Engine (retrieve, score, pack)
  semantic.py        contract + policies + lineage
  store.py           DuckDB warehouse + governed queries
  investigation.py   narratives + citation guard
  war_room.py        multi-objective decision + action payload
  evaluation.py      measured harness + calibration
  scripts.py         CLI: build | demo
tests/               automated verification
requirements.txt
.env.example
```

---

## Design choices

- **Four analytical methods** as the core stack: STL + Isolation Forest, ETS
  baseline, contribution vs planted truth (SHAP-style), Price–Volume–Mix.
- **Dual materiality:** statistically unusual **and** commercially meaningful.
- **Interaction residual** is always shown; contributions are not normalized to 100%.
- **RBAC inside retrieval** (filter before rank), not a post-hoc UI hide.
- **Policies cited by ID** (e.g. P018, P031) — decision rights come from the
  policy KB, not a hardcoded threshold.
- **KPI definition conflict** (ERP net revenue vs marketing gross booked) is
  surfaced in lineage, not silently overwritten.
- **Graceful degradation:** no API keys / no optional vector DB → still runs and
  stays honest.

---

