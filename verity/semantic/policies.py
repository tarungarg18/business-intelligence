"""Business Policy Knowledge Base and decision-rights resolution.

Two jobs:

  1. Expose policies as retrievable documents, so the Evidence Engine can cite
     them by ID alongside incident evidence.
  2. Answer "who is allowed to approve this?" from a *retrieved* policy rather
     than a hardcoded constant.

An owner is not an authority. A recommendation to cut price by 12% has an
owner (the regional marketing head) and, separately, an approver (the CFO,
because 12% exceeds the 10% regional limit in P018). Conflating the two is the
gap this module closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


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

        Policies enter retrieval through the same contract as incident
        documents, so a policy citation is indistinguishable in form from an
        operations-report citation.
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

        Returns the governing policy, the limit it imposes, and who must sign
        off if the proposal exceeds it.
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
    return {"percent": "%", "percentage_points": "pp", "inr": " INR", "units": " units"}.get(
        unit, ""
    )


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
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[1] / "configs" / "policies.yaml"
    )
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
