"""Feedback capture for continuous learning loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class FeedbackEntry:
    scenario_id: str
    user_id: str
    verdict: str
    reason: str
    chosen_lens: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackStore:
    def __init__(self) -> None:
        self._entries: list[FeedbackEntry] = []

    def add(self, entry: FeedbackEntry) -> None:
        self._entries.append(entry)

    @property
    def entries(self) -> tuple[FeedbackEntry, ...]:
        return tuple(self._entries)

    def lens_preferences(self) -> dict[str, float]:
        counts: dict[str, int] = {}
        total = 0
        for entry in self._entries:
            if entry.chosen_lens:
                counts[entry.chosen_lens] = counts.get(entry.chosen_lens, 0) + 1
                total += 1
        return {lens: count / total for lens, count in counts.items()} if total else {}
