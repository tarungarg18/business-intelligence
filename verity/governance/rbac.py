"""Identity, entitlements, and access decisions.

Every query and every retrieval carries a ``Principal``. Access is resolved
from the semantic contract, never from a hardcoded rule, and every decision is
recorded whether it was allowed or denied.

The important property: the same Principal governs both structured queries and
document retrieval. That is what makes "the AI cannot reason over data the
human cannot see" a structural guarantee rather than a policy statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from verity.semantic import KPIContract, SemanticContract


@dataclass(frozen=True)
class Principal:
    """Who is asking, and what they are entitled to."""

    user_id: str
    role: str
    region: str | None = None

    @property
    def attributes(self) -> Mapping[str, Any]:
        attrs: dict[str, Any] = {}
        if self.region:
            attrs["region"] = self.region
        return attrs

    def describe(self) -> str:
        scope = f", region={self.region}" if self.region else ""
        return f"{self.user_id} (role={self.role}{scope})"


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    principal: Principal
    resource: str
    row_filter: str | None
    reason: str

    @property
    def result(self) -> str:
        return "ALLOWED" if self.allowed else "DENIED"


def authorize_kpi(
    contract: SemanticContract | KPIContract,
    kpi_name: str | None,
    principal: Principal,
    requested_region: str | None = None,
) -> AccessDecision:
    """Decide whether ``principal`` may read a KPI, and under what row filter.

    Two distinct denials are possible and they are reported differently:

      * the role has no entitlement to the KPI at all (column-level denial)
      * the role is entitled, but asked for rows outside its scope (row-level)
    """
    kpi = contract if isinstance(contract, KPIContract) else contract[kpi_name or ""]
    resource = f"{kpi.name}" + (f".region={requested_region}" if requested_region else "")

    if not kpi.access.permits(principal.role):
        return AccessDecision(
            allowed=False,
            principal=principal,
            resource=resource,
            row_filter=None,
            reason=(
                f"role {principal.role!r} is not granted access to KPI "
                f"{kpi.name!r} (permitted: {list(kpi.access.roles)})"
            ),
        )

    row_filter = kpi.access.filter_for(principal.role, principal.attributes)

    if requested_region and principal.region and requested_region != principal.region:
        if row_filter is not None:
            return AccessDecision(
                allowed=False,
                principal=principal,
                resource=resource,
                row_filter=row_filter,
                reason=(
                    f"ROW_LEVEL_POLICY: {principal.role!r} is scoped to "
                    f"{principal.region!r} and may not read {requested_region!r}"
                ),
            )

    return AccessDecision(
        allowed=True,
        principal=principal,
        resource=resource,
        row_filter=row_filter,
        reason=(
            f"role {principal.role!r} permitted"
            + (f"; row filter applied: {row_filter}" if row_filter else "; unrestricted")
        ),
    )


def visible_documents(documents, principal: Principal):
    """Filter a document iterable to what ``principal`` may retrieve.

    Applied BEFORE ranking, not after. An unauthorised document is never a
    retrieval candidate, so it cannot influence scores or reach the LLM.
    """
    allowed = []
    for doc in documents:
        roles = getattr(doc, "access_roles", None)
        if roles is None:
            meta = getattr(doc, "metadata", {}) or {}
            raw = meta.get("access_roles", "")
            roles = tuple(r for r in str(raw).split(",") if r)
        if principal.role in roles:
            allowed.append(doc)
    return allowed


# Standard demo principals, referenced by the scenarios and the UI.
DEMO_PRINCIPALS: Mapping[str, Principal] = {
    "cfo": Principal(user_id="exec_04", role="executive"),
    "analyst": Principal(user_id="analyst_11", role="analyst"),
    "west_manager": Principal(user_id="regional_manager_17", role="regional_manager", region="West"),
    "east_manager": Principal(user_id="regional_manager_23", role="regional_manager", region="East"),
}
