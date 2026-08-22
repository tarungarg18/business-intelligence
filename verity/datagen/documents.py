"""Unstructured evidence corpus: operations reports, ticket clusters, news, notes.

Every document carries the full metadata contract, because retrieval filters on
metadata *before* ranking. In particular ``access_roles`` is what makes RBAC
reach into retrieval itself: a document the user may not see is never a
candidate, so a War Room agent cannot reason over it either.

The corpus deliberately contains distractors — documents that are semantically
close but belong to another region or period. Retrieval quality is only
measurable if wrong answers are actually available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

ALL_ROLES = ("executive", "analyst", "regional_manager")
STAFF_ROLES = ("executive", "analyst")


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    document_type: str
    title: str
    text: str
    timestamp: date
    region: str | None = None
    product: str | None = None
    kpi: str | None = None
    source_reliability: float = 0.8
    access_roles: tuple[str, ...] = ALL_ROLES

    def to_metadata(self) -> dict[str, Any]:
        """Metadata contract used for filtering before ranking."""
        return {
            "source": self.source,
            "document_type": self.document_type,
            "region": self.region or "",
            "product": self.product or "",
            "kpi": self.kpi or "",
            "timestamp": self.timestamp.isoformat(),
            "source_reliability": self.source_reliability,
            "access_roles": ",".join(self.access_roles),
        }


# --------------------------------------------------------------------------
# S1 — West multi-factor revenue shock (the main demo path)
# --------------------------------------------------------------------------

_S1 = [
    Document(
        id="E1042",
        source="operations_report",
        document_type="incident_report",
        title="Warehouse W3 dispatch backlog — West",
        text=(
            "Warehouse W3 experienced a 14-hour dispatch backlog beginning 11 August "
            "following a conveyor fault on the primary outbound line. Approximately "
            "1,900 confirmed orders were held past their committed dispatch window. "
            "Priority SKUs in the Home category were most affected. Stock cover for "
            "West priority SKUs fell below seven days during the incident. Normal "
            "dispatch throughput resumed on 15 August after the line was repaired."
        ),
        timestamp=date(2026, 8, 11),
        region="West",
        kpi="net_revenue",
        source_reliability=0.94,
    ),
    Document(
        id="E1055",
        source="support_ticket_cluster",
        document_type="ticket_cluster",
        title="Delivery delay complaints spike — West (214 tickets)",
        text=(
            "Cluster of 214 support tickets raised between 11 and 15 August, all "
            "citing delayed delivery against a committed date in the West region. "
            "Sixty-one tickets escalated to cancellation requests. Ticket volume for "
            "this reason code is approximately 6.2x the trailing eight-week average. "
            "Agents repeatedly reference warehouse dispatch as the stated cause."
        ),
        timestamp=date(2026, 8, 13),
        region="West",
        kpi="net_revenue",
        source_reliability=0.88,
    ),
    Document(
        id="E1061",
        source="news",
        document_type="market_news",
        title="Competitor announces regional pricing campaign",
        text=(
            "A competing manufacturer announced an aggressive regional pricing "
            "campaign across western metros effective 9 August, positioning entry "
            "Home-category units roughly 8 percent below prevailing market price. "
            "Trade coverage suggests the campaign is a share-capture move ahead of "
            "the festive quarter. No confirmed volume impact has been published."
        ),
        timestamp=date(2026, 8, 9),
        region="West",
        kpi="net_revenue",
        source_reliability=0.62,
    ),
    Document(
        id="E1067",
        source="internal_note",
        document_type="planning_note",
        title="West promotion calendar gap — August",
        text=(
            "The West mid-season Home promotion concluded on 7 August and was not "
            "renewed for the following cycle pending budget review. No replacement "
            "offer was scheduled. Historically this promotion has carried "
            "approximately 9 percent of weekly Home-category volume in West."
        ),
        timestamp=date(2026, 8, 8),
        region="West",
        kpi="net_revenue",
        source_reliability=0.90,
    ),
]


# --------------------------------------------------------------------------
# S2 — North dip with contradictory evidence (abstention must trigger)
# --------------------------------------------------------------------------

_S2 = [
    Document(
        id="E1112",
        source="marketing_report",
        document_type="campaign_report",
        title="North campaign performance review (period ending 28 June)",
        text=(
            "The North regional campaign delivered against plan for the period "
            "ending 28 June, with reach and engagement both slightly ahead of "
            "target. No performance concerns were raised. Note: this report covers "
            "the June cycle and has not been refreshed for July."
        ),
        timestamp=date(2026, 6, 28),
        region="North",
        kpi="net_revenue",
        source_reliability=0.55,
    ),
    Document(
        id="E1118",
        source="support_ticket_cluster",
        document_type="ticket_cluster",
        title="North complaint volume increase (48 tickets)",
        text=(
            "Support ticket volume in North rose by roughly 30 percent in the week "
            "of 20 July. Reason codes are dispersed across product quality, billing, "
            "and delivery with no dominant category. Agents did not identify a "
            "common root cause."
        ),
        timestamp=date(2026, 7, 22),
        region="North",
        kpi="net_revenue",
        source_reliability=0.83,
    ),
    Document(
        id="E1124",
        source="operations_report",
        document_type="incident_report",
        title="North operations review — no incident identified",
        text=(
            "A review of North fulfilment operations for the week of 20 July found "
            "no service incident, no dispatch backlog, and no stock availability "
            "constraint. Dispatch reliability was within normal bounds at 98.1 "
            "percent. Operations does not consider fulfilment a contributing factor "
            "to the revenue movement."
        ),
        timestamp=date(2026, 7, 23),
        region="North",
        kpi="net_revenue",
        source_reliability=0.90,
    ),
    Document(
        id="E1130",
        source="news",
        document_type="market_news",
        title="Analysts note broad softness in northern markets",
        text=(
            "Sector commentary published in mid-July referenced general demand "
            "softness across northern markets without attributing a specific cause "
            "or quantifying the effect. The commentary predates the period in "
            "question and cites no primary data."
        ),
        timestamp=date(2026, 7, 19),
        region="North",
        kpi="net_revenue",
        source_reliability=0.40,
    ),
]


# --------------------------------------------------------------------------
# S3 — Sparse-history launch
# --------------------------------------------------------------------------

_S3 = [
    Document(
        id="E1201",
        source="internal_note",
        document_type="planning_note",
        title="Ember Nano launch promotion — West",
        text=(
            "The Ember Nano (SKU-E) launch promotion ran in West from 3 August to 9 "
            "August with introductory pricing and paid placement. Launch-period "
            "volume should not be treated as a representative baseline; the product "
            "has fewer than six weeks of trading history."
        ),
        timestamp=date(2026, 8, 3),
        region="West",
        product="SKU-E",
        kpi="units_sold",
        source_reliability=0.90,
    ),
]


# --------------------------------------------------------------------------
# Distractors — semantically similar, wrong region or wrong period.
# Without these, Precision@K is meaningless.
# --------------------------------------------------------------------------

_DISTRACTORS = [
    Document(
        id="E0910",
        source="operations_report",
        document_type="incident_report",
        title="Warehouse E1 dispatch delay — East",
        text=(
            "Warehouse E1 recorded a 6-hour dispatch delay on 3 June following a "
            "staffing shortfall on the evening shift. Roughly 240 orders were "
            "affected. Throughput normalised the following day with no further "
            "escalation."
        ),
        timestamp=date(2026, 6, 3),
        region="East",
        kpi="net_revenue",
        source_reliability=0.92,
    ),
    Document(
        id="E0934",
        source="support_ticket_cluster",
        document_type="ticket_cluster",
        title="Delivery complaints — West (historical, March)",
        text=(
            "A cluster of 88 delivery-delay tickets was recorded in West during "
            "March following a regional transport strike. The matter was closed and "
            "service levels recovered within eight days."
        ),
        timestamp=date(2026, 3, 14),
        region="West",
        kpi="net_revenue",
        source_reliability=0.86,
    ),
    Document(
        id="E0951",
        source="news",
        document_type="market_news",
        title="Competitor pricing action in eastern markets",
        text=(
            "A competitor introduced promotional pricing across eastern metros in "
            "May, discounting Outdoor-category units. Market response was reported "
            "as muted."
        ),
        timestamp=date(2026, 5, 21),
        region="East",
        kpi="net_revenue",
        source_reliability=0.60,
    ),
    Document(
        id="E0968",
        source="internal_note",
        document_type="planning_note",
        title="North promotion calendar — Q2 review",
        text=(
            "North promotional activity for the second quarter concluded as planned. "
            "Budget for the following cycle was approved without change. No calendar "
            "gaps were identified."
        ),
        timestamp=date(2026, 6, 30),
        region="North",
        kpi="net_revenue",
        source_reliability=0.88,
    ),
    Document(
        id="E0977",
        source="operations_report",
        document_type="incident_report",
        title="West inventory count variance — February",
        text=(
            "A routine cycle count in West identified a minor inventory variance of "
            "0.4 percent across Home-category SKUs in February. The variance was "
            "reconciled and no availability impact occurred."
        ),
        timestamp=date(2026, 2, 17),
        region="West",
        kpi="units_sold",
        source_reliability=0.91,
    ),
    Document(
        id="E0989",
        source="hr_note",
        document_type="internal_memo",
        title="West depot shift roster change",
        text=(
            "The West depot moved to a revised three-shift roster from 1 August to "
            "improve evening dispatch coverage. Headcount is unchanged. This is a "
            "scheduling change only and carries confidential staffing detail."
        ),
        timestamp=date(2026, 8, 1),
        region="West",
        kpi=None,
        source_reliability=0.75,
        # Restricted: a regional manager must not retrieve this. Used to prove
        # that entitlement filtering happens inside retrieval.
        access_roles=STAFF_ROLES,
    ),
]


DOCUMENTS: tuple[Document, ...] = tuple(_S1 + _S2 + _S3 + _DISTRACTORS)

DOCUMENT_BY_ID = {d.id: d for d in DOCUMENTS}


def documents_for_role(role: str) -> tuple[Document, ...]:
    return tuple(d for d in DOCUMENTS if role in d.access_roles)
