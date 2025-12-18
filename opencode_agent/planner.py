from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PlanStep:
    description: str
    status: str = "pending"  # pending, in_progress, done, blocked


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)

    def add(self, description: str) -> None:
        self.steps.append(PlanStep(description=description))

    def start(self, index: int) -> None:
        self._validate_index(index)
        self.steps[index].status = "in_progress"

    def complete(self, index: int) -> None:
        self._validate_index(index)
        self.steps[index].status = "done"

    def block(self, index: int) -> None:
        self._validate_index(index)
        self.steps[index].status = "blocked"

    def to_text(self) -> str:
        lines = ["Planned steps:"]
        if not self.steps:
            lines.append("(no steps; add with `plan add <text>`)")
        for i, step in enumerate(self.steps, start=1):
            lines.append(f"{i}. [{step.status}] {step.description}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.steps.clear()

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= len(self.steps):
            raise IndexError("plan step index out of range")
