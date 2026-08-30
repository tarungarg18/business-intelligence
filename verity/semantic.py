"""Governed semantic layer: KPI contracts, business policies, knowledge graph.

This module is the only place permitted to answer three questions:

    * What does a KPI mean, and where does it come from?
    * Is a given movement material, and who may see it?
    * Who is allowed to approve a proposed action?

Everything else asks this module rather than hardcoding an answer. The contract
and policy knowledge base are loaded and validated from ``configs/``; the lineage
graph is a view over the same contract, so definition and visualisation are one
artifact rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Materiality is a dual gate: a movement must be both statistically notable and
# commercially meaningful. These are the recognised impact units.
IMPACT_METRICS = {"absolute_inr", "absolute_units", "percentage_points"}
VALID_CADENCES = {"daily", "weekly", "monthly", "irregular"}


# --------------------------------------------------------------------------- #
# KPI semantic contract
# --------------------------------------------------------------------------- #


class ContractError(ValueError):
    """Raised when the semantic contract is malformed."""


@dataclass(frozen=True)
class Source:
    name: str
    type: str
    cadence: str
    freshness_sla_hours: int
    authoritative: bool = False

    def is_stale(self, age_hours: float) -> bool:
        return age_hours > self.freshness_sla_hours


@dataclass(frozen=True)
class Materiality:
    warning_pct: float
    critical_pct: float
    impact_metric: str
    minimum_business_impact: float

    def severity(self, change_pct: float) -> str:
        """Statistical half of the gate only. Use ``KPIContract.assess``."""
        magnitude = abs(change_pct)
        if magnitude >= self.critical_pct:
            return "critical"
        if magnitude >= self.warning_pct:
            return "warning"
        return "normal"


@dataclass(frozen=True)
class Access:
    roles: tuple[str, ...]
    row_filter: Mapping[str, str] = field(default_factory=dict)

    def permits(self, role: str) -> bool:
        return role in self.roles

    def filter_for(self, role: str, user_attrs: Mapping[str, Any]) -> str | None:
        """Return a concrete row filter for ``role``, or None if unrestricted.

        The contract stores filters as templates like ``region = user.region``.
        This substitutes the caller's own attributes, so a West manager can never
        be handed an East filter.
        """
        template = self.row_filter.get(role)
        if not template:
            return None
        rendered = template
        for key, value in user_attrs.items():
            rendered = rendered.replace(f"user.{key}", repr(value))
        if "user." in rendered:
            missing = rendered.split("user.", 1)[1].split()[0]
            raise ContractError(
                f"row filter for role {role!r} references unknown user attribute "
                f"'user.{missing}'"
            )
        return rendered


@dataclass(frozen=True)
class MaterialityVerdict:
    """Outcome of the dual gate, carrying its own explanation."""

    is_material: bool
    severity: str
    change_pct: float
    business_impact: float
    impact_metric: str
    reason: str

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.is_material


@dataclass(frozen=True)
class KPIContract:
    name: str
    label: str
    formula: str
    unit: str
    grain: str
    description: str
    sources: tuple[Source, ...]
    dimensions: tuple[str, ...]
    materiality: Materiality
    drivers: tuple[str, ...]
    ownership: Mapping[str, str]
    access: Access
    lineage: tuple[str, ...]

    @property
    def authoritative_source(self) -> Source:
        for source in self.sources:
            if source.authoritative:
                return source
        raise ContractError(f"KPI {self.name!r} has no authoritative source")

    def source(self, name: str) -> Source:
        for candidate in self.sources:
            if candidate.name == name:
                return candidate
        raise ContractError(f"KPI {self.name!r} has no source named {name!r}")

    def assess(self, change_pct: float, business_impact: float) -> MaterialityVerdict:
        """Apply the dual materiality gate.

        A movement is material only when it clears the statistical threshold and
        the minimum business impact. Either alone is insufficient: a 20% swing on
        a trivial base is noise, and a large absolute number that is statistically
        ordinary is business-as-usual.
        """
        severity = self.materiality.severity(change_pct)
        clears_statistical = severity != "normal"
        clears_business = abs(business_impact) >= self.materiality.minimum_business_impact

        if clears_statistical and clears_business:
            reason = (
                f"{abs(change_pct):.1f}% movement clears the "
                f"{severity} threshold ({getattr(self.materiality, severity + '_pct'):.1f}%), "
                f"and business impact {abs(business_impact):,.0f} clears the minimum "
                f"{self.materiality.minimum_business_impact:,.0f} "
                f"({self.materiality.impact_metric})"
            )
            return MaterialityVerdict(
                True, severity, change_pct, business_impact,
                self.materiality.impact_metric, reason,
            )

        if clears_statistical and not clears_business:
            reason = (
                f"{abs(change_pct):.1f}% movement is statistically notable but "
                f"business impact {abs(business_impact):,.0f} is below the minimum "
                f"{self.materiality.minimum_business_impact:,.0f} "
                f"({self.materiality.impact_metric}) — suppressed to avoid alert fatigue"
            )
        elif clears_business and not clears_statistical:
            reason = (
                f"business impact {abs(business_impact):,.0f} is large but the "
                f"{abs(change_pct):.1f}% movement is within normal variation "
                f"(warning threshold {self.materiality.warning_pct:.1f}%)"
            )
        else:
            reason = (
                f"{abs(change_pct):.1f}% movement is within normal variation and "
                f"business impact {abs(business_impact):,.0f} is below the minimum"
            )
        return MaterialityVerdict(
            False, "normal", change_pct, business_impact,
            self.materiality.impact_metric, reason,
        )


@dataclass(frozen=True)
class KPIRelationship:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class DefinitionConflict:
    kpi: str
    authoritative_source: str
    conflicting_source: str
    authoritative_definition: str
    conflicting_definition: str
    reason: str


@dataclass(frozen=True)
class SemanticContract:
    kpis: Mapping[str, KPIContract]
    relationships: tuple[KPIRelationship, ...]
    conflicts: tuple[DefinitionConflict, ...]

    def __getitem__(self, name: str) -> KPIContract:
        try:
            return self.kpis[name]
        except KeyError:
            raise ContractError(
                f"unknown KPI {name!r}; contract defines {sorted(self.kpis)}"
            ) from None

    def __iter__(self) -> Iterator[KPIContract]:
        return iter(self.kpis.values())

    def __len__(self) -> int:
        return len(self.kpis)

    def visible_to(self, role: str) -> tuple[KPIContract, ...]:
        return tuple(k for k in self.kpis.values() if k.access.permits(role))

    def conflicts_for(self, kpi: str) -> tuple[DefinitionConflict, ...]:
        return tuple(c for c in self.conflicts if c.kpi == kpi)

    def downstream_of(self, kpi: str) -> tuple[KPIRelationship, ...]:
        return tuple(r for r in self.relationships if r.source == kpi)


def _require(node: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in node:
        raise ContractError(f"{where}: missing required key {key!r}")
    return node[key]


def _parse_source(node: Mapping[str, Any], kpi_name: str) -> Source:
    where = f"kpi {kpi_name!r} source"
    name = _require(node, "name", where)
    cadence = _require(node, "cadence", f"{where} {name!r}")
    if cadence not in VALID_CADENCES:
        raise ContractError(
            f"{where} {name!r}: cadence {cadence!r} not in {sorted(VALID_CADENCES)}"
        )
    return Source(
        name=name,
        type=_require(node, "type", f"{where} {name!r}"),
        cadence=cadence,
        freshness_sla_hours=int(_require(node, "freshness_sla_hours", f"{where} {name!r}")),
        authoritative=bool(node.get("authoritative", False)),
    )


def _parse_kpi(node: Mapping[str, Any]) -> KPIContract:
    name = _require(node, "name", "kpi")
    where = f"kpi {name!r}"

    definition = _require(node, "definition", where)
    materiality_node = _require(node, "materiality", where)
    access_node = node.get("access", {})

    impact_metric = _require(materiality_node, "impact_metric", f"{where} materiality")
    if impact_metric not in IMPACT_METRICS:
        raise ContractError(
            f"{where}: impact_metric {impact_metric!r} not in {sorted(IMPACT_METRICS)}"
        )

    materiality = Materiality(
        warning_pct=float(_require(materiality_node, "warning_pct", f"{where} materiality")),
        critical_pct=float(_require(materiality_node, "critical_pct", f"{where} materiality")),
        impact_metric=impact_metric,
        minimum_business_impact=float(
            _require(materiality_node, "minimum_business_impact", f"{where} materiality")
        ),
    )
    if materiality.warning_pct >= materiality.critical_pct:
        raise ContractError(
            f"{where}: warning_pct ({materiality.warning_pct}) must be below "
            f"critical_pct ({materiality.critical_pct})"
        )

    sources = tuple(_parse_source(s, name) for s in _require(node, "sources", where))
    if not any(s.authoritative for s in sources):
        raise ContractError(f"{where}: no source is marked authoritative")

    access = Access(
        roles=tuple(access_node.get("roles", ())),
        row_filter=dict(access_node.get("row_filter") or {}),
    )
    for filtered_role in access.row_filter:
        if filtered_role not in access.roles:
            raise ContractError(
                f"{where}: row_filter defined for role {filtered_role!r}, which is "
                f"not in the permitted roles {list(access.roles)}"
            )

    return KPIContract(
        name=name,
        label=node.get("label", name),
        formula=_require(definition, "formula", f"{where} definition"),
        unit=_require(definition, "unit", f"{where} definition"),
        grain=_require(definition, "grain", f"{where} definition"),
        description=(definition.get("description") or "").strip(),
        sources=sources,
        dimensions=tuple(node.get("dimensions", ())),
        materiality=materiality,
        drivers=tuple(_require(node, "drivers", where)),
        ownership=dict(node.get("ownership") or {}),
        access=access,
        lineage=tuple(node.get("lineage", ())),
    )


def load_contract(path: str | Path | None = None) -> SemanticContract:
    """Load and validate the KPI semantic contract."""
    path = Path(path) if path else CONFIG_DIR / "kpis.yaml"
    if not path.exists():
        raise ContractError(f"semantic contract not found at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kpi_nodes: Sequence[Mapping[str, Any]] = raw.get("kpis") or []
    if not kpi_nodes:
        raise ContractError(f"{path}: contract defines no KPIs")

    kpis: dict[str, KPIContract] = {}
    for node in kpi_nodes:
        contract = _parse_kpi(node)
        if contract.name in kpis:
            raise ContractError(f"duplicate KPI definition {contract.name!r}")
        kpis[contract.name] = contract

    relationships = tuple(
        KPIRelationship(
            source=_require(r, "from", "kpi_relationships"),
            target=_require(r, "to", "kpi_relationships"),
            relation=r.get("relation", "related"),
        )
        for r in raw.get("kpi_relationships") or []
    )
    for rel in relationships:
        for endpoint in (rel.source, rel.target):
            if endpoint not in kpis:
                raise ContractError(f"kpi_relationships references unknown KPI {endpoint!r}")

    conflicts = tuple(
        DefinitionConflict(
            kpi=_require(c, "kpi", "definition_conflicts"),
            authoritative_source=_require(c, "authoritative_source", "definition_conflicts"),
            conflicting_source=_require(c, "conflicting_source", "definition_conflicts"),
            authoritative_definition=c.get("authoritative_definition", ""),
            conflicting_definition=c.get("conflicting_definition", ""),
            reason=c.get("reason", ""),
        )
        for c in raw.get("definition_conflicts") or []
    )
    for conflict in conflicts:
        if conflict.kpi not in kpis:
            raise ContractError(f"definition_conflicts references unknown KPI {conflict.kpi!r}")
        # Both named sources must actually be declared on that KPI, or the
        # reconciliation shown in the UI would cite a source that does not exist.
        kpis[conflict.kpi].source(conflict.authoritative_source)
        kpis[conflict.kpi].source(conflict.conflicting_source)

    return SemanticContract(kpis=kpis, relationships=relationships, conflicts=conflicts)


# --------------------------------------------------------------------------- #
# Business policy knowledge base and decision rights
# --------------------------------------------------------------------------- #


class PolicyError(ValueError):
    """Raised when the policy knowledge base is malformed."""


@dataclass(frozen=True)
class Authority:
    role: str
    max_value: float
    unit: str
    escalation_role: str
    hard_ceiling: float | None = None
    ceiling_escalation_role: str | None = None


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    lever: str
    text: str
    authority: Authority
    source_reliability: float
    access_roles: tuple[str, ...]

    def as_evidence(self) -> dict[str, Any]:
        """Render as an Evidence Pack entry.

        Policies enter retrieval through the same contract as incident documents,
        so a policy citation is indistinguishable in form from an operations
        report citation.
        """
        return {
            "id": self.id,
            "source": "policy_db",
            "type": "policy",
            "reliability": self.source_reliability,
            "text": " ".join(self.text.split()),
            "access_roles": list(self.access_roles),
            "lever": self.lever,
        }


@dataclass(frozen=True)
class AuthorityVerdict:
    """Whether a proposed action falls within someone's decision rights."""

    lever: str
    proposed_value: float
    unit: str
    within_authority: bool
    owner_role: str
    required_approval: str | None
    policy_id: str
    limit: float
    rationale: str

    @property
    def status(self) -> str:
        return "within_authority" if self.within_authority else "escalation_required"


@dataclass(frozen=True)
class PolicyBook:
    policies: Mapping[str, Policy]
    role_hierarchy: Mapping[str, int]

    def __getitem__(self, policy_id: str) -> Policy:
        try:
            return self.policies[policy_id]
        except KeyError:
            raise PolicyError(
                f"unknown policy {policy_id!r}; known: {sorted(self.policies)}"
            ) from None

    def __iter__(self) -> Iterator[Policy]:
        return iter(self.policies.values())

    def __len__(self) -> int:
        return len(self.policies)

    def for_lever(self, lever: str) -> Policy:
        for policy in self.policies.values():
            if policy.lever == lever:
                return policy
        raise PolicyError(
            f"no policy governs lever {lever!r}; "
            f"known levers: {sorted(p.lever for p in self.policies.values())}"
        )

    def visible_to(self, role: str) -> tuple[Policy, ...]:
        return tuple(p for p in self.policies.values() if role in p.access_roles)

    def check_authority(
        self, lever: str, proposed_value: float, acting_role: str
    ) -> AuthorityVerdict:
        """Resolve decision rights for a proposed action.

        Returns the governing policy, the limit it imposes, and who must sign off
        if the proposal exceeds it.
        """
        policy = self.for_lever(lever)
        authority = policy.authority
        magnitude = abs(proposed_value)

        ceiling = authority.hard_ceiling
        if ceiling is not None and magnitude > ceiling:
            return AuthorityVerdict(
                lever=lever,
                proposed_value=proposed_value,
                unit=authority.unit,
                within_authority=False,
                owner_role=authority.role,
                required_approval=authority.ceiling_escalation_role,
                policy_id=policy.id,
                limit=ceiling,
                rationale=(
                    f"{magnitude:g}{_unit_suffix(authority.unit)} exceeds the hard "
                    f"ceiling of {ceiling:g}{_unit_suffix(authority.unit)} in "
                    f"{policy.id} ({policy.title})"
                ),
            )

        if magnitude <= authority.max_value:
            # Even within the limit, the acting role must actually hold the right.
            holds_right = self._outranks(acting_role, authority.role)
            return AuthorityVerdict(
                lever=lever,
                proposed_value=proposed_value,
                unit=authority.unit,
                within_authority=holds_right,
                owner_role=authority.role,
                required_approval=None if holds_right else authority.role,
                policy_id=policy.id,
                limit=authority.max_value,
                rationale=(
                    f"{magnitude:g}{_unit_suffix(authority.unit)} is within the "
                    f"{authority.max_value:g}{_unit_suffix(authority.unit)} limit "
                    f"granted to {authority.role} by {policy.id}"
                    if holds_right
                    else (
                        f"{magnitude:g}{_unit_suffix(authority.unit)} is within the "
                        f"{authority.role} limit, but {acting_role} does not hold "
                        f"that right under {policy.id}"
                    )
                ),
            )

        return AuthorityVerdict(
            lever=lever,
            proposed_value=proposed_value,
            unit=authority.unit,
            within_authority=False,
            owner_role=authority.role,
            required_approval=authority.escalation_role,
            policy_id=policy.id,
            limit=authority.max_value,
            rationale=(
                f"{magnitude:g}{_unit_suffix(authority.unit)} exceeds the "
                f"{authority.max_value:g}{_unit_suffix(authority.unit)} limit for "
                f"{authority.role} under {policy.id} ({policy.title}); "
                f"{authority.escalation_role} approval required"
            ),
        )

    def _outranks(self, role: str, required: str) -> bool:
        have = self.role_hierarchy.get(role)
        need = self.role_hierarchy.get(required)
        if have is None or need is None:
            return False
        return have >= need


def _unit_suffix(unit: str) -> str:
    return {"percent": "%", "percentage_points": "pp", "inr": " INR", "units": " units"}.get(unit, "")


def _parse_authority(node: Mapping[str, Any], policy_id: str) -> Authority:
    where = f"policy {policy_id!r} authority"
    for key in ("role", "max_value", "unit", "escalation_role"):
        if key not in node:
            raise PolicyError(f"{where}: missing required key {key!r}")
    return Authority(
        role=node["role"],
        max_value=float(node["max_value"]),
        unit=node["unit"],
        escalation_role=node["escalation_role"],
        hard_ceiling=float(node["hard_ceiling"]) if "hard_ceiling" in node else None,
        ceiling_escalation_role=node.get("ceiling_escalation_role"),
    )


def load_policies(path: str | Path | None = None) -> PolicyBook:
    """Load and validate the business policy knowledge base."""
    path = Path(path) if path else CONFIG_DIR / "policies.yaml"
    if not path.exists():
        raise PolicyError(f"policy knowledge base not found at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    nodes = raw.get("policies") or []
    if not nodes:
        raise PolicyError(f"{path}: no policies defined")

    hierarchy = dict(raw.get("role_hierarchy") or {})
    policies: dict[str, Policy] = {}
    levers: dict[str, str] = {}

    for node in nodes:
        for key in ("id", "title", "lever", "text", "authority"):
            if key not in node:
                raise PolicyError(f"policy {node.get('id', '?')!r}: missing key {key!r}")

        policy_id = node["id"]
        if policy_id in policies:
            raise PolicyError(f"duplicate policy id {policy_id!r}")

        lever = node["lever"]
        if lever in levers:
            raise PolicyError(
                f"policies {levers[lever]!r} and {policy_id!r} both govern lever "
                f"{lever!r}; authority resolution would be ambiguous"
            )
        levers[lever] = policy_id

        authority = _parse_authority(node["authority"], policy_id)
        for role in (authority.role, authority.escalation_role, authority.ceiling_escalation_role):
            if role and role not in hierarchy:
                raise PolicyError(
                    f"policy {policy_id!r} references role {role!r} which is absent "
                    f"from role_hierarchy"
                )

        policies[policy_id] = Policy(
            id=policy_id,
            title=node["title"],
            lever=lever,
            text=node["text"],
            authority=authority,
            source_reliability=float(node.get("source_reliability", 1.0)),
            access_roles=tuple(node.get("access_roles", ())),
        )

    return PolicyBook(policies=policies, role_hierarchy=hierarchy)


# --------------------------------------------------------------------------- #
# KPI lineage graph (a view over the contract)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LineageNode:
    id: str
    label: str
    kind: str


@dataclass(frozen=True)
class LineageEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class LineageGraph:
    kpi: str
    nodes: tuple[LineageNode, ...]
    edges: tuple[LineageEdge, ...]
    conflicts: tuple[dict[str, str], ...]

    def as_dict(self) -> dict:
        return {
            "kpi": self.kpi,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
            "conflicts": list(self.conflicts),
        }


def build_lineage_graph(contract: SemanticContract, kpi_name: str) -> LineageGraph:
    kpi = contract[kpi_name]
    nodes: list[LineageNode] = [
        LineageNode(f"kpi:{kpi.name}", kpi.label, "kpi"),
        LineageNode(f"formula:{kpi.name}", kpi.formula, "formula"),
    ]
    edges = [LineageEdge(f"formula:{kpi.name}", f"kpi:{kpi.name}", "calculates")]

    previous = f"formula:{kpi.name}"
    for item in reversed(kpi.lineage):
        nodes.append(LineageNode(item, item, _lineage_kind(item)))
        edges.append(LineageEdge(item, previous, "feeds"))
        previous = item

    for source in kpi.sources:
        source_id = f"source:{source.name}"
        nodes.append(LineageNode(source_id, f"{source.name} ({source.cadence})", "source"))
        for lineage_item in kpi.lineage:
            if lineage_item.startswith(source.name + "."):
                edges.append(LineageEdge(source_id, lineage_item, "provides"))

    for relationship in contract.downstream_of(kpi.name):
        target = contract[relationship.target]
        nodes.append(LineageNode(f"kpi:{target.name}", target.label, "related_kpi"))
        edges.append(LineageEdge(f"kpi:{kpi.name}", f"kpi:{target.name}", relationship.relation))

    return LineageGraph(
        kpi=kpi.name,
        nodes=_dedupe_nodes(nodes),
        edges=tuple(edges),
        conflicts=tuple(_conflict_dict(c) for c in contract.conflicts_for(kpi.name)),
    )


def _lineage_kind(item: str) -> str:
    if item.startswith("kpi."):
        return "kpi"
    if item.startswith("transform."):
        return "transform"
    return "field"


def _dedupe_nodes(nodes: list[LineageNode]) -> tuple[LineageNode, ...]:
    seen: dict[str, LineageNode] = {}
    for node in nodes:
        seen.setdefault(node.id, node)
    return tuple(seen.values())


def _conflict_dict(conflict: DefinitionConflict) -> dict[str, str]:
    return {
        "kpi": conflict.kpi,
        "authoritative_source": conflict.authoritative_source,
        "conflicting_source": conflict.conflicting_source,
        "authoritative_definition": conflict.authoritative_definition,
        "conflicting_definition": conflict.conflicting_definition,
        "reason": conflict.reason,
    }
