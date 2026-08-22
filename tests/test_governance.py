"""Entitlements, the governed query path, and the audit trail."""

from __future__ import annotations

from datetime import date

import pytest

from verity.datagen import DOCUMENTS, generate
from verity.governance import DEMO_PRINCIPALS, Principal, visible_documents
from verity.governance.audit import AuditLog
from verity.semantic import load_contract
from verity.store import AccessDenied, Warehouse, validate_registry


@pytest.fixture(scope="module")
def contract():
    return load_contract()


@pytest.fixture(scope="module")
def warehouse(contract):
    """In-memory warehouse so tests never touch the on-disk build."""
    wh = Warehouse(contract, path=None, audit=AuditLog())
    wh.build(generate(), DOCUMENTS)
    yield wh
    wh.close()


WEST = DEMO_PRINCIPALS["west_manager"]
ANALYST = DEMO_PRINCIPALS["analyst"]
WINDOW = {"start": date(2026, 8, 10), "end": date(2026, 8, 16)}


# --- registry -------------------------------------------------------------


def test_sql_registry_covers_the_contract(contract):
    validate_registry(contract)  # raises if the two have drifted


# --- row-level security ---------------------------------------------------


def test_manager_can_read_own_region(warehouse):
    frame = warehouse.kpi_series("net_revenue", WEST, **WINDOW)
    assert not frame.empty


def test_manager_cannot_read_another_region(warehouse):
    with pytest.raises(AccessDenied, match="ROW_LEVEL_POLICY"):
        warehouse.kpi_series("net_revenue", WEST, region="East")


def test_row_filter_applies_even_when_no_region_is_requested(warehouse):
    """The caller must not be able to widen their own scope by omitting a filter."""
    scoped = warehouse.kpi_series("net_revenue", WEST, **WINDOW)["value"].sum()
    everything = warehouse.sql(
        "SELECT SUM(net_revenue) v FROM erp_sales "
        "WHERE date BETWEEN '2026-08-10' AND '2026-08-16'"
    )["v"][0]
    assert scoped < everything


def test_two_managers_see_disjoint_slices(warehouse):
    west = warehouse.kpi_series("net_revenue", WEST, **WINDOW)["value"].sum()
    east = warehouse.kpi_series(
        "net_revenue", DEMO_PRINCIPALS["east_manager"], **WINDOW
    )["value"].sum()
    assert west != east


# --- column-level security ------------------------------------------------


def test_manager_cannot_read_margin(warehouse):
    with pytest.raises(AccessDenied, match="not granted access"):
        warehouse.kpi_series("gross_margin_pct", WEST)


def test_analyst_can_read_margin(warehouse):
    assert not warehouse.kpi_series("gross_margin_pct", ANALYST, **WINDOW).empty


def test_analyst_sees_every_region(warehouse):
    frame = warehouse.kpi_series("net_revenue", ANALYST, group_by=("region",), **WINDOW)
    assert set(frame["region"]) == {"West", "East", "North"}


# --- retrieval-level security --------------------------------------------


def test_restricted_document_is_filtered_before_ranking(warehouse):
    """RBAC reaches into retrieval: the document is never a candidate at all."""
    manager_docs = set(warehouse.documents(WEST)["id"])
    analyst_docs = set(warehouse.documents(ANALYST)["id"])
    assert "E0989" in analyst_docs
    assert "E0989" not in manager_docs


def test_visible_documents_helper_agrees_with_the_warehouse(warehouse):
    helper = {d.id for d in visible_documents(DOCUMENTS, WEST)}
    stored = set(warehouse.documents(WEST)["id"])
    assert helper == stored


# --- auditing -------------------------------------------------------------


def test_denials_are_audited(contract):
    audit = AuditLog()
    wh = Warehouse(contract, path=None, audit=audit)
    wh.build(generate(), DOCUMENTS)
    with pytest.raises(AccessDenied):
        wh.kpi_series("net_revenue", WEST, region="East")
    assert len(audit.denials()) == 1
    entry = audit.denials()[0]
    assert entry.user_id == WEST.user_id
    assert entry.resource == "net_revenue.region=East"
    assert entry.result == "DENIED"
    wh.close()


def test_successful_reads_are_audited_too(contract):
    """A log that records only denials cannot answer 'who saw this?'."""
    audit = AuditLog()
    wh = Warehouse(contract, path=None, audit=audit)
    wh.build(generate(), DOCUMENTS)
    wh.kpi_series("net_revenue", ANALYST, **WINDOW)
    assert len(audit) == 1
    assert audit.entries[0].result == "ALLOWED"
    wh.close()


def test_warehouse_uses_the_audit_log_it_was_given(contract):
    """Regression: an empty AuditLog is falsy by length, so `audit or AuditLog()`
    silently swapped in a different log and the caller's stayed empty."""
    audit = AuditLog()
    assert not len(audit)
    assert bool(audit) is True, "an empty log must not be falsy"

    wh = Warehouse(contract, path=None, audit=audit)
    wh.build(generate(), DOCUMENTS)
    wh.kpi_series("net_revenue", ANALYST, **WINDOW)
    assert len(audit) == 1, "warehouse wrote to a different log than the caller holds"
    wh.close()


def test_audit_entries_render_for_the_viewer(contract):
    audit = AuditLog()
    wh = Warehouse(contract, path=None, audit=audit)
    wh.build(generate(), DOCUMENTS)
    with pytest.raises(AccessDenied):
        wh.kpi_series("net_revenue", WEST, region="East")
    rendered = audit.denials()[0].render()
    for field in ("timestamp:", "user:", "resource:", "action:", "result:", "reason:"):
        assert field in rendered
    wh.close()


# --- query guards ---------------------------------------------------------


def test_unknown_group_by_is_rejected(warehouse):
    with pytest.raises(ValueError, match="cannot group"):
        warehouse.kpi_series("net_revenue", ANALYST, group_by=("warehouse_id",))


def test_unsupported_frequency_is_rejected(warehouse):
    with pytest.raises(ValueError, match="unsupported frequency"):
        warehouse.kpi_series("net_revenue", ANALYST, freq="fortnight")
