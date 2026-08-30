"""Semantic contract, materiality gate, and decision rights."""

from __future__ import annotations

import pytest

from verity.semantic import load_contract, load_policies
from verity.semantic import ContractError, PolicyError


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def policies():
    return load_policies()


# --- contract -------------------------------------------------------------


def test_contract_loads_expected_kpis(contract):
    assert set(contract.kpis) == {
        "net_revenue",
        "gross_margin_pct",
        "units_sold",
        "customer_churn",
    }


def test_every_kpi_has_an_authoritative_source(contract):
    for kpi in contract:
        assert kpi.authoritative_source is not None


def test_unknown_kpi_raises(contract):
    with pytest.raises(ContractError, match="unknown KPI"):
        contract["revenue"]


def test_lineage_terminates_at_the_kpi(contract):
    for kpi in contract:
        if kpi.lineage:
            assert kpi.lineage[-1] == f"kpi.{kpi.name}"


# --- dual materiality gate ------------------------------------------------


def test_material_when_both_thresholds_clear(contract):
    verdict = contract["net_revenue"].assess(change_pct=-12.0, business_impact=4_200_000)
    assert verdict.is_material
    assert verdict.severity == "critical"


def test_not_material_when_business_impact_too_small(contract):
    """A large percentage on a trivial base is noise, not a signal."""
    verdict = contract["net_revenue"].assess(change_pct=-12.0, business_impact=90_000)
    assert not verdict.is_material
    assert "below the minimum" in verdict.reason


def test_not_material_when_movement_is_ordinary(contract):
    """A big absolute number that is statistically unremarkable is business as usual."""
    verdict = contract["net_revenue"].assess(change_pct=-1.2, business_impact=8_000_000)
    assert not verdict.is_material
    assert "within normal variation" in verdict.reason


def test_warning_and_critical_bands_are_distinct(contract):
    kpi = contract["net_revenue"]
    assert kpi.assess(-6.0, 10_000_000).severity == "warning"
    assert kpi.assess(-11.0, 10_000_000).severity == "critical"


def test_gate_is_symmetric_for_positive_movements(contract):
    kpi = contract["net_revenue"]
    assert kpi.assess(12.0, 4_200_000).is_material
    assert kpi.assess(-12.0, -4_200_000).is_material


# --- access ---------------------------------------------------------------


def test_margin_is_hidden_from_regional_managers(contract):
    assert not contract["gross_margin_pct"].access.permits("regional_manager")
    assert contract["net_revenue"].access.permits("regional_manager")


def test_row_filter_binds_to_the_callers_own_region(contract):
    access = contract["net_revenue"].access
    assert access.filter_for("regional_manager", {"region": "West"}) == "region = 'West'"
    assert access.filter_for("regional_manager", {"region": "East"}) == "region = 'East'"


def test_unfiltered_roles_get_no_filter(contract):
    assert contract["net_revenue"].access.filter_for("analyst", {"region": "West"}) is None


def test_row_filter_rejects_missing_user_attribute(contract):
    with pytest.raises(ContractError, match="unknown user attribute"):
        contract["net_revenue"].access.filter_for("regional_manager", {})


# --- decision rights ------------------------------------------------------


def test_within_authority(policies):
    verdict = policies.check_authority("discount_pct", 8.0, "regional_manager")
    assert verdict.within_authority
    assert verdict.required_approval is None
    assert verdict.policy_id == "P018"


def test_exceeding_limit_escalates_to_cfo(policies):
    verdict = policies.check_authority("discount_pct", 12.0, "regional_manager")
    assert not verdict.within_authority
    assert verdict.required_approval == "cfo"
    assert verdict.status == "escalation_required"


def test_hard_ceiling_escalates_beyond_cfo(policies):
    verdict = policies.check_authority("discount_pct", 25.0, "cfo")
    assert not verdict.within_authority
    assert verdict.required_approval == "cfo_and_cro"


def test_authority_limit_comes_from_a_citable_policy(policies):
    """The limit must be traceable to a retrievable document, not a constant."""
    verdict = policies.check_authority("discount_pct", 12.0, "regional_manager")
    policy = policies[verdict.policy_id]
    assert policy.as_evidence()["type"] == "policy"
    assert str(verdict.limit) in policy.text or "10" in policy.text


def test_unknown_lever_raises(policies):
    with pytest.raises(PolicyError, match="no policy governs lever"):
        policies.check_authority("teleportation", 1.0, "cfo")


def test_every_policy_is_retrievable_as_evidence(policies):
    for policy in policies:
        evidence = policy.as_evidence()
        assert evidence["id"] == policy.id
        assert evidence["source"] == "policy_db"
        assert evidence["text"]
