"""FastAPI boundary for Verity."""

from __future__ import annotations

from fastapi import FastAPI, Query

from verity.app.service import VerityDemoService
from verity.governance import DEMO_PRINCIPALS

app = FastAPI(
    title="Verity KPI Intelligence-to-Action Engine",
    version="0.1.0",
    description="Deterministic analytics, governed evidence, War Room decisions.",
)

service = VerityDemoService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scenarios": len(service.scenarios())}


@app.get("/scenarios")
def scenarios() -> list[dict]:
    return service.scenarios()


@app.get("/scenario/{scenario_id}")
def scenario(
    scenario_id: str,
    principal: str = Query("analyst", pattern="^(cfo|analyst|west_manager|east_manager)$"),
    persona: str = Query("analyst", pattern="^(cfo|analyst|ops)$"),
) -> dict:
    return service.run_scenario(
        scenario_id,
        principal=DEMO_PRINCIPALS[principal],
        persona=persona,
    ).as_dict()


@app.get("/lineage/{kpi}")
def lineage(kpi: str = "net_revenue") -> dict:
    return service.lineage(kpi).as_dict()


@app.get("/evaluation")
def evaluation() -> dict:
    report = service.evaluation()
    return {"metrics": report.as_dict(), "scenarios": [row.__dict__ for row in report.scenarios]}


@app.get("/model-health")
def model_health() -> dict:
    return service.model_health()


@app.post("/audit/exercise-denial")
def exercise_denial() -> dict:
    return {"result": service.exercise_denial(), "audit": service.audit_view()}


@app.get("/audit")
def audit() -> list[dict]:
    return service.audit_view()


@app.get("/risk-radar")
def risk_radar() -> list[dict]:
    return service.risk_radar()
