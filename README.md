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
| RBAC / entitlements | 🚧 in progress |
| DuckDB warehouse | 🚧 in progress |
| Analytical engine (STL, IF, SHAP, ETS, PVM) | ⬜ planned |
| Evidence Engine (hybrid Graph-RAG) | ⬜ planned |
| Decision War Room | ⬜ planned |
| Evaluation harness | ⬜ planned |

---

## Setup

```bash
pip install -r requirements.txt
```

From the repository root:

```bash
# Generate the synthetic world and build the warehouse
python -m verity.scripts.build_data
```

All commands assume the repository root is on `PYTHONPATH` (running from the
root satisfies this).

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
