"""KPI lineage graph rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass

from verity.semantic.contract import KPIContract, SemanticContract


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
        kind = _kind(item)
        nodes.append(LineageNode(item, item, kind))
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


def _kind(item: str) -> str:
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


def _conflict_dict(conflict) -> dict[str, str]:
    return {
        "kpi": conflict.kpi,
        "authoritative_source": conflict.authoritative_source,
        "conflicting_source": conflict.conflicting_source,
        "authoritative_definition": conflict.authoritative_definition,
        "conflicting_definition": conflict.conflicting_definition,
        "reason": conflict.reason,
    }
