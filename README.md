# Verity - KPI Intelligence-to-Action Engine

Accenture Innovation Challenge 2026 - Problem Track 3: BusinessIntelligence.ai
Team Verity Exchange

Verity detects material KPI movements, explains them from governed evidence,
resolves competing business objectives into a recommended action, and abstains
when the evidence does not support a conclusion.

The central product rule:

> The LLM is never the source of quantitative truth.

Detection, attribution, forecasting, retrieval, ranking, scoring, permissions,
lineage, policy checks and simulation are deterministic. The LLM layer receives
an Evidence Pack and deterministic outputs only.

## Status

Working prototype, implemented as a deterministic local demo.

| Component | State |
|---|---|
| KPI semantic contract + validation | working |
| Business policy KB + decision rights | working |
| Synthetic world + recorded ground truth | working |
| RBAC / entitlements / audit | working |
| DuckDB warehouse + governed queries | working |
| Detection (STL + Isolation Forest) | working |
| Baseline forecast (ETS + intervals) | working |
| Provider-agnostic LLM/embedding layer | working |
| Attribution + Price-Volume-Mix | working |
| Evidence Engine + deterministic reranking | working |
| Persona narratives + citation guard | working |
| Decision War Room + action payload | working |
| Cost governor (tiered routing) | working |
| Evaluation harness + model health | working |
| FastAPI + Streamlit demo surfaces | working |

83 tests passing with `python -m pytest`.

## Setup

```bash
pip install -r requirements.txt

# Generate the synthetic world and build the warehouse
python -m verity.scripts build

# Run the deterministic CLI demo
python -m verity.scripts demo

# Run the API
uvicorn verity.app.api:app --reload

# Run the demo UI
streamlit run verity/app/ui.py
```

API entry points:

| Endpoint | Purpose |
|---|---|
| `/health` | service readiness |
| `/scenarios` | available synthetic scenarios |
| `/scenario/S1` | full signal to evidence to War Room to action payload |
| `/lineage/net_revenue` | KPI lineage and definition conflict |
| `/evaluation` | measured detection, retrieval and abstention metrics |
| `/model-health` | drift and review panel values |
| `/audit` | audit log |
| `/risk-radar` | scenario replay risk table |

## API Keys

Every key is optional. With no hosted provider configured, Verity still runs
using offline LSA embeddings and a deterministic template generator.

Copy `.env.example` to `.env` and set any of:

```text
GEMINI_API_KEY=...      # primary embeddings + generation
OPENROUTER_API_KEY=...  # generation fallback
HF_TOKEN=...            # Hugging Face generation fallback
```

The generation chain is Gemini, OpenRouter/OpenAI-compatible, Hugging Face,
then offline template. OpenRouter and Hugging Face are chat-only fallbacks here;
they are skipped for embeddings.

## Synthetic Scenarios

| Scenario | Movement | Planted truth | Expected behaviour |
|---|---|---|---|
| S1 West multi-factor | about -12% | inventory, promotion, competitor activity | explain and decide |
| S2 North contradiction | about -9% | cause is not recoverable from evidence | abstain |
| S3 SKU-E launch | +22% | promotion with sparse history | explain with lower confidence |
| S4 East control | about 0% | no shock | stay silent |

S1 carries an explicit interaction residual. Verity displays that residual
instead of forcing contributions to sum to 100%.

## Layout

```text
verity/
  analytics/         STL, Isolation Forest, ETS, attribution, PVM, what-if
  app/               FastAPI and Streamlit surfaces
  configs/           KPI semantic contract and policy KB
  datagen/           Synthetic world, ground truth, evidence corpus
  governance/        RBAC, audit, cost governor
  llm/               Provider abstraction and fallback chain
  rag/               Evidence Engine and Evidence Pack builder
  semantic.py        KPI contracts, policy loading, lineage graph
  store.py           DuckDB warehouse and governed query path
  investigation.py   Persona narratives and citation verification
  war_room.py        Two-round decision synthesis and action payload
  evaluation.py      Measured scenario metrics and calibration
  scripts.py         CLI: build the warehouse, run the demo
```

## Design Notes

Only four analytical methods are presented as the core stack: STL + Isolation
Forest, SHAP-style contribution against planted synthetic truth, ETS forecast,
and Price-Volume-Mix decomposition.

Retrieval-level RBAC filters documents before ranking. A War Room agent cannot
reason over evidence that the requesting user is not allowed to retrieve.

Policy documents are retrieved and cited by ID, so decision-rights escalation
comes from the policy KB rather than a hardcoded threshold.
