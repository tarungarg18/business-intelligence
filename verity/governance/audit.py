"""Append-only audit log.

Every access decision is recorded — allowed or denied. A log that only records
denials cannot answer "who saw this?", which is the question that actually
matters in a governance review.

The log is deliberately append-only: there is no update or delete path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from verity.governance.rbac import AccessDecision, Principal

AUDIT_COLUMNS = (
    "timestamp",
    "user_id",
    "role",
    "region",
    "resource",
    "action",
    "result",
    "reason",
    "row_filter",
    "detail",
)


@dataclass(frozen=True)
class AuditEntry:
    timestamp: datetime
    user_id: str
    role: str
    region: str | None
    resource: str
    action: str
    result: str
    reason: str
    row_filter: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "role": self.role,
            "region": self.region or "",
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "reason": self.reason,
            "row_filter": self.row_filter or "",
            "detail": json.dumps(self.detail, default=str, sort_keys=True),
        }

    def render(self) -> str:
        """Human-readable form, as shown in the audit viewer."""
        lines = [
            f"timestamp: {self.timestamp:%Y-%m-%d %H:%M:%S}",
            f"user:      {self.user_id}",
            f"resource:  {self.resource}",
            f"action:    {self.action}",
            f"result:    {self.result}",
        ]
        if self.result == "DENIED":
            lines.append(f"reason:    {self.reason}")
        return "\n".join(lines)


class AuditLog:
    """In-memory append-only log, optionally mirrored to JSONL on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._path = Path(path) if path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # A log object always exists, even when it holds nothing yet. Without
        # this, `__len__` makes an empty log falsy and idioms like
        # `audit or AuditLog()` quietly substitute a different log.
        return True

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def record(
        self,
        principal: Principal,
        resource: str,
        action: str,
        result: str,
        reason: str = "",
        row_filter: str | None = None,
        **detail: Any,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc),
            user_id=principal.user_id,
            role=principal.role,
            region=principal.region,
            resource=resource,
            action=action,
            result=result,
            reason=reason,
            row_filter=row_filter,
            detail=detail,
        )
        self._entries.append(entry)
        if self._path:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.as_row(), default=str) + "\n")
        return entry

    def record_decision(
        self, decision: AccessDecision, action: str = "READ", **detail: Any
    ) -> AuditEntry:
        return self.record(
            principal=decision.principal,
            resource=decision.resource,
            action=action,
            result=decision.result,
            reason=decision.reason,
            row_filter=decision.row_filter,
            **detail,
        )

    def denials(self) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self._entries if e.result == "DENIED")

    def for_user(self, user_id: str) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self._entries if e.user_id == user_id)

    def to_rows(self) -> list[dict[str, Any]]:
        return [e.as_row() for e in self._entries]
