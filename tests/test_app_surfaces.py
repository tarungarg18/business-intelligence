"""Application and API surfaces."""

from __future__ import annotations

from fastapi.testclient import TestClient

from verity.app.api import app
from verity.app.service import VerityDemoService


def test_service_runs_main_demo_and_exposes_visible_surfaces():
    service = VerityDemoService()
    try:
        bundle = service.run_scenario("S1")
        payload = bundle.as_dict()
        assert payload["assessment"]["detected"]
        assert payload["evidence_pack"]["evidence"]
        assert payload["decision"]["action_payload"]["status"] == "DISPATCHED"

        lineage = service.lineage("net_revenue").as_dict()
        assert lineage["nodes"]
        assert lineage["edges"]
        assert lineage["conflicts"]

        health = service.model_health()
        assert health["false_alarm_rate"] == 0.0
        assert health["review_trigger"] == "none"

        denial = service.exercise_denial()
        assert "ROW_LEVEL_POLICY" in denial
        assert any(row["result"] == "DENIED" for row in service.audit_view())
    finally:
        service.close()


def test_api_exposes_scenario_lineage_and_evaluation():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    scenario = client.get(
        "/scenario/S1",
        params={"principal": "west_manager", "persona": "ops"},
    ).json()
    assert scenario["decision"]["rounds"] == 2
    assert scenario["evidence_pack"]["confidence"] > 0

    lineage = client.get("/lineage/net_revenue").json()
    assert lineage["conflicts"][0]["conflicting_source"] == "promotion_api"

    evaluation = client.get("/evaluation").json()
    assert evaluation["metrics"]["driver_top1_accuracy"] == 1.0
