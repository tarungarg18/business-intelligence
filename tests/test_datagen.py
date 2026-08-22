"""Synthetic world: determinism, ground truth, and the planted scenarios."""

from __future__ import annotations

import pytest

from verity.datagen import DOCUMENT_BY_ID, DOCUMENTS, SCENARIO_BY_ID, generate
from verity.datagen.entities import (
    EXPECTED_ABSTAIN,
    EXPECTED_EXPLAIN,
    EXPECTED_SILENT,
)
from verity.semantic import load_contract


@pytest.fixture(scope="module")
def data():
    return generate()


@pytest.fixture(scope="module")
def truth(data):
    return data.ground_truth


# --- determinism ----------------------------------------------------------


def test_generation_is_deterministic(data):
    again = generate()
    assert data.erp_sales.equals(again.erp_sales)
    assert data.ground_truth.equals(again.ground_truth)


def test_different_seeds_produce_different_worlds(data):
    other = generate(seed=1234)
    assert not data.erp_sales.equals(other.erp_sales)


# --- source shape ---------------------------------------------------------


def test_sources_have_different_grains(data):
    """Reconciling mismatched grain is part of what the engine must prove."""
    assert "date" in data.erp_sales.columns          # daily
    assert "week_start" in data.promotion_api.columns  # weekly
    assert "week_start" in data.crm_accounts.columns   # weekly


def test_promotion_api_reports_gross_not_net(data):
    """The marketing feed must genuinely disagree with ERP, or the KPI-conflict
    scenario has nothing to reconcile."""
    erp_net = data.erp_sales["net_revenue"].sum()
    api_gross = data.promotion_api["gross_booked_value"].sum()
    assert api_gross > erp_net
    # Difference is returns + freight, roughly 3.5%.
    assert 0.02 < (api_gross - erp_net) / erp_net < 0.06


def test_new_product_has_no_rows_before_launch(data):
    from verity.datagen.entities import PRODUCT_BY_SKU

    launch = PRODUCT_BY_SKU["SKU-E"].launch_date
    rows = data.erp_sales[data.erp_sales["product"] == "SKU-E"]
    assert rows["date"].min() >= launch


# --- ground truth ---------------------------------------------------------


def test_every_scenario_is_represented(truth):
    assert set(truth["scenario_id"]) == set(SCENARIO_BY_ID)


def test_s1_is_a_multi_factor_shock_near_minus_twelve_percent(truth):
    s1 = truth[truth["scenario_id"] == "S1"]
    assert len(s1) == 3, "S1 must have three distinct planted drivers"
    assert s1.iloc[0]["total_movement_pct"] == pytest.approx(-12.0, abs=0.5)


def test_s1_drivers_are_ranked_by_true_contribution(truth):
    s1 = truth[truth["scenario_id"] == "S1"].sort_values("rank")
    assert list(s1["driver"]) == ["inventory", "promotion", "competitor_activity"]
    magnitudes = [abs(v) for v in s1["true_contribution_pp"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_s1_has_a_genuine_interaction_residual(truth):
    """Contributions must NOT sum to the total. The gap is real interaction,
    and it is the ceiling on what any attribution method can recover."""
    s1 = truth[truth["scenario_id"] == "S1"]
    residual = s1.iloc[0]["interaction_residual_pp"]
    assert residual != 0.0
    contributions = s1["true_contribution_pp"].sum()
    total = s1.iloc[0]["total_movement_pct"]
    assert total == pytest.approx(contributions + residual, abs=1e-6)


def test_control_scenario_plants_nothing(truth):
    s4 = truth[truth["scenario_id"] == "S4"]
    assert s4.iloc[0]["driver"] is None
    assert s4.iloc[0]["expected_behaviour"] == EXPECTED_SILENT


def test_expected_behaviours_cover_explain_and_abstain(truth):
    behaviours = set(truth["expected_behaviour"])
    assert EXPECTED_EXPLAIN in behaviours
    assert EXPECTED_ABSTAIN in behaviours


def test_ground_truth_drivers_exist_in_the_contract():
    """Ground truth and the semantic contract must not drift apart."""
    contract = load_contract()
    for scenario in SCENARIO_BY_ID.values():
        declared = set(contract[scenario.kpi].drivers)
        for factor in scenario.factors:
            assert factor.driver in declared, (
                f"{scenario.id} plants driver {factor.driver!r} which is not "
                f"declared on KPI {scenario.kpi!r}"
            )


# --- document corpus ------------------------------------------------------


def test_relevant_documents_exist():
    for scenario in SCENARIO_BY_ID.values():
        for doc_id in scenario.relevant_doc_ids:
            assert doc_id in DOCUMENT_BY_ID, f"{scenario.id} cites missing doc {doc_id}"


def test_corpus_contains_distractors():
    """Precision@K is meaningless without retrievable wrong answers."""
    relevant = {
        doc_id
        for scenario in SCENARIO_BY_ID.values()
        for doc_id in scenario.relevant_doc_ids
    }
    distractors = [d for d in DOCUMENTS if d.id not in relevant]
    assert len(distractors) >= 5


def test_s2_evidence_genuinely_contradicts():
    """The abstention scenario needs real disagreement, not just weak evidence."""
    tickets = DOCUMENT_BY_ID["E1118"]   # complaints rose
    ops = DOCUMENT_BY_ID["E1124"]       # no incident found
    assert tickets.region == ops.region == "North"
    assert "no service incident" in ops.text
    assert "rose" in tickets.text


def test_restricted_document_is_marked_staff_only():
    hr_note = DOCUMENT_BY_ID["E0989"]
    assert "regional_manager" not in hr_note.access_roles
    assert "analyst" in hr_note.access_roles


def test_every_document_carries_the_metadata_contract():
    required = {
        "source",
        "document_type",
        "region",
        "product",
        "kpi",
        "timestamp",
        "source_reliability",
        "access_roles",
    }
    for doc in DOCUMENTS:
        assert required <= set(doc.to_metadata())
