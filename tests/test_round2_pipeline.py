"""Round 2 plan implementation: evidence, action and evaluation."""

from __future__ import annotations

import pytest

from verity.analytics import assess_window, attribute_window, price_volume_mix, simulate_action
from verity.datagen import DOCUMENTS, SCENARIO_BY_ID, generate
from verity.evaluation import evaluate_scenarios
from verity.governance import DEMO_PRINCIPALS, route_for_assessment
from verity.investigation import render_narrative, verify_citations
from verity.rag import build_evidence_pack, retrieve_evidence
from verity.scripts.demo import run_demo
from verity.semantic import load_contract
from verity.store import Warehouse
from verity.war_room import convene_war_room


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def warehouse(contract):
    wh = Warehouse(contract, path=None)
    wh.build(generate(), DOCUMENTS)
    yield wh
    wh.close()


def _assessment(warehouse, contract, scenario_id):
    scenario = SCENARIO_BY_ID[scenario_id]
    principal = DEMO_PRINCIPALS["analyst"]
    if scenario.products:
        frame = warehouse.sql(
            "SELECT date AS period, SUM(units) AS value FROM erp_sales "
            "WHERE region = ? AND product = ? GROUP BY date ORDER BY date",
            [scenario.region, scenario.products[0]],
        )
    else:
        frame = warehouse.kpi_series(scenario.kpi, principal, region=scenario.region)
    return assess_window(frame, contract[scenario.kpi], scenario.window_start, scenario.window_end)


def test_attribution_keeps_unexplained_residual_visible(warehouse):
    s1 = SCENARIO_BY_ID["S1"]
    result = attribute_window(
        warehouse,
        kpi=s1.kpi,
        region=s1.region,
        start=s1.window_start,
        end=s1.window_end,
        scenario_id=s1.id,
    )
    assert [c.driver for c in result.contributions] == [
        "inventory",
        "promotion",
        "competitor_activity",
    ]
    assert result.unexplained_residual_pp == pytest.approx(0.4501)
    assert result.total_movement_pct == pytest.approx(
        result.explained_pp + result.unexplained_residual_pp
    )


def test_pvm_is_deterministic_arithmetic(warehouse):
    s1 = SCENARIO_BY_ID["S1"]
    result = price_volume_mix(warehouse, region=s1.region, start=s1.window_start, end=s1.window_end)
    assert result.method == "price_volume_mix"
    assert result.total_pp != 0


def test_retrieval_filters_before_ranking_and_includes_policy(warehouse):
    west = DEMO_PRINCIPALS["west_manager"]
    evidence = retrieve_evidence(
        warehouse,
        west,
        query="West warehouse dispatch backlog approval policy",
        kpi="net_revenue",
        region="West",
        top_k=8,
    )
    ids = {item.id for item in evidence}
    assert "E0989" not in ids
    assert "E1042" in ids
    assert any(item.id.startswith("P") for item in evidence)


def test_contradictory_scenario_abstains(warehouse, contract):
    s2 = SCENARIO_BY_ID["S2"]
    assessment = _assessment(warehouse, contract, "S2")
    attribution = attribute_window(
        warehouse,
        kpi=s2.kpi,
        region=s2.region,
        start=s2.window_start,
        end=s2.window_end,
        scenario_id=s2.id,
    )
    pack = build_evidence_pack(
        warehouse,
        DEMO_PRINCIPALS["analyst"],
        assessment=assessment,
        attribution=attribution,
    )
    assert pack.contradictions
    assert pack.should_abstain


def test_sparse_history_lowers_confidence(warehouse, contract):
    s3 = SCENARIO_BY_ID["S3"]
    assessment = _assessment(warehouse, contract, "S3")
    attribution = attribute_window(
        warehouse,
        kpi=s3.kpi,
        region=s3.region,
        start=s3.window_start,
        end=s3.window_end,
        scenario_id=s3.id,
    )
    pack = build_evidence_pack(
        warehouse,
        DEMO_PRINCIPALS["analyst"],
        assessment=assessment,
        attribution=attribution,
    )
    assert not assessment.sufficient_history
    assert pack.confidence <= 0.55


def test_war_room_is_bounded_and_emits_action_payload(warehouse, contract):
    s1 = SCENARIO_BY_ID["S1"]
    assessment = _assessment(warehouse, contract, "S1")
    attribution = attribute_window(
        warehouse,
        kpi=s1.kpi,
        region=s1.region,
        start=s1.window_start,
        end=s1.window_end,
        scenario_id=s1.id,
    )
    pack = build_evidence_pack(
        warehouse,
        DEMO_PRINCIPALS["west_manager"],
        assessment=assessment,
        attribution=attribution,
        query="inventory warehouse dispatch approval policy",
    )
    decision = convene_war_room(pack, DEMO_PRINCIPALS["west_manager"])
    assert decision.rounds == 2
    assert decision.converged
    assert decision.action_payload.action_id.startswith("ACT-")
    assert decision.action_payload.policy_id.startswith("P")
    assert decision.simulation.expected_revenue_effect_pp > 0


def test_narrative_citations_are_from_pack(warehouse, contract):
    s1 = SCENARIO_BY_ID["S1"]
    assessment = _assessment(warehouse, contract, "S1")
    attribution = attribute_window(
        warehouse,
        kpi=s1.kpi,
        region=s1.region,
        start=s1.window_start,
        end=s1.window_end,
        scenario_id=s1.id,
    )
    pack = build_evidence_pack(
        warehouse,
        DEMO_PRINCIPALS["analyst"],
        assessment=assessment,
        attribution=attribution,
    )
    narrative = render_narrative(pack)
    assert not verify_citations(narrative.summary, pack)
    assert verify_citations("Invented citation E9999 and 12345 units", pack)


def test_cost_governor_routes_war_room_only_for_critical(warehouse, contract):
    s1_route = route_for_assessment(_assessment(warehouse, contract, "S1"))
    s4_route = route_for_assessment(_assessment(warehouse, contract, "S4"))
    assert s1_route.tier == 3
    assert s1_route.war_room_allowed
    assert s4_route.tier == 0
    assert not s4_route.war_room_allowed


def test_evaluation_reports_measured_numbers(warehouse):
    report = evaluate_scenarios(warehouse, DEMO_PRINCIPALS["analyst"])
    metrics = report.as_dict()
    assert metrics["materiality_recall"] == pytest.approx(1.0)
    assert metrics["false_alarm_rate"] == pytest.approx(0.0)
    assert metrics["driver_top1_accuracy"] == pytest.approx(1.0)
    assert metrics["recall_at_k"] > 0.5


def test_demo_path_reaches_decision_payload():
    result = run_demo("S1")
    assert result["decision"] is not None
    assert result["decision"].action_payload.status == "DISPATCHED"
    assert result["evidence_pack"].evidence


def test_what_if_numbers_are_deterministic():
    first = simulate_action("discount_pct", 10)
    second = simulate_action("discount_pct", 10)
    assert first == second
    assert first.expected_revenue_effect_pp == pytest.approx(5.8)
