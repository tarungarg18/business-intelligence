# Verity — KPI Intelligence-to-Action Engine

**Accenture Innovation Challenge 2026 — Problem Track 3 (BusinessIntelligence.ai)**
Team Verity Exchange · IIT Guwahati

Verity detects material KPI movements, explains them from governed evidence,
resolves competing business objectives into a recommended action, and says so
plainly when the evidence does not support a conclusion.

The architectural commitment running through the whole system:

> **The LLM is never the source of quantitative truth.**
> Detection, attribution, forecasting, retrieval, ranking, scoring, and
> permissions are all deterministic. The LLM receives an Evidence Pack and
> nothing else.

The full design document (`Round2_Solution_Plan.md`) is kept outside this
repository alongside the challenge source material.

---

## Status

Under active development against a four-week build plan.

| Component | State |
|---|---|
| KPI semantic contract + validation | ✅ working |
| Business policy KB + decision rights | ✅ working |
| Synthetic world + recorded ground truth | ✅ working |
| RBAC / entitlements / audit | ✅ working |
| DuckDB warehouse + governed queries | ✅ working |
| Detection (STL + Isolation Forest) | ✅ working |
| Baseline forecast (ETS + intervals) | ✅ working |
| Provider-agnostic LLM/embedding layer | ✅ working |
| Attribution (SHAP, Price–Volume–Mix) | 🚧 next |
| Evidence Engine (hybrid Graph-RAG) | 🚧 next |
| Decision War Room | ⬜ planned |
| Evaluation harness | ⬜ planned |

66 tests passing (`python -m pytest`).

---

## Setup

```bash
pip install -r requirements.txt

# Generate the synthetic world and build the warehouse
python -m verity.scripts.build_data
```

Run from the repository root; that puts `verity` on the import path.

### API keys are optional

Copy `.env.example` to `.env` and add a key to use hosted embeddings and
generation:

```
GEMINI_API_KEY=...      # primary
OPENCODE_API_KEY=...    # fallback, any OpenAI-compatible endpoint
```

With **no key at all** the system still runs end to end. It falls back to
TF-IDF + SVD (LSA) embeddings via scikit-learn and a template generator, and
says so in the UI rather than pretending otherwise.

There is deliberately **no local model download**. `sentence-transformers` was
the original plan but pulls in roughly 2 GB of torch to embed a few hundred
chunks. Embeddings come from an API over plain REST and are cached to disk, so
after the first build the demo runs offline — which matters more for a live
pitch than model quality does.

---

## Why the data is synthetic — and why that is a feature

The world in `verity/datagen/` is generated from a fixed seed, and the
scenarios are planted deliberately. Each shock runs three ways: a
counterfactual with no shock, the actual, and one isolated run per causal
factor. Differencing those yields the **true** contribution of every driver.

That means accuracy can be *measured* rather than asserted:

| Scenario | Movement | Planted truth | Correct behaviour |
|---|---|---|---|
| **S1** West multi-factor | −12.05% | inventory −7.00 pp · promotion −3.50 pp · competitor −2.00 pp | identify and rank the drivers |
| **S2** North dip | −9.00% | cause not recoverable from evidence | **abstain** |
| **S3** New SKU-E | +22.00% | promotion, but <6 weeks history | explain with reduced confidence |
| **S4** East control | ~0% | nothing planted | **stay silent** (false-alarm check) |

S1 also carries a **+0.45 pp interaction residual** — real interaction between
the three factors that no single-factor attribution can recover. Verity
displays it as unexplained rather than normalising contributions to 100%.

---

## Layout

```
verity/
├── configs/        KPI semantic contract, business policy KB
├── semantic/       Contract loading, validation, decision rights
├── datagen/        Synthetic world + ground truth + document corpus
├── governance/     RBAC, audit, cost governor
├── store/          DuckDB warehouse
├── analytics/      STL · Isolation Forest · SHAP · ETS · Price-Volume-Mix
├── rag/            Evidence Engine: ingest → chunk → embed → retrieve → rerank
├── investigation/  Hypothesis generation and verification
├── war_room/       Multi-objective decision synthesis
├── evaluation/     Measured accuracy, calibration, drift
└── scripts/        Build and demo entry points
```

---

## Design notes

**Four analytical methods, not seven.** STL + Isolation Forest for detection,
SHAP for attribution, ETS for the expected baseline, Price–Volume–Mix for
financial decomposition. Granger causality and Prophet were considered and
rejected — a short list every team member can defend beats a long list nobody
can.

**Contributions never sum to 100%.** Where attribution falls short of the
observed movement, the shortfall is shown as an unexplained residual.

**RBAC lives inside retrieval.** Documents carry `access_roles`, and filtering
happens *before* ranking. An unauthorised document is never a candidate, so a
War Room agent cannot reason over it — the guarantee is structural, not a
policy statement.
