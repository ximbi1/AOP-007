from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


STATE_FILE = ".opencode/state.json"


@dataclass
class Evaluation:
    status: str = "UNKNOWN"  # SUCCESS | PARTIAL | FAILURE | UNKNOWN
    evidence: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ProjectState:
    root: Path
    languages: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    key_files: List[str] = field(default_factory=list)
    directories: List[str] = field(default_factory=list)
    central_files: List[str] = field(default_factory=list)
    semantic_summaries: Dict[str, str] = field(default_factory=dict)
    project_understanding: str = ""
    project_understanding_structured: Dict[str, str] = field(default_factory=dict)
    files_modified: List[str] = field(default_factory=list)
    modification_summaries: Dict[str, str] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    evaluation: Evaluation = field(default_factory=Evaluation)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, root: Path) -> "ProjectState":
        path = root / STATE_FILE
        if not path.exists():
            return cls(root=root)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(root=root)
        return cls(
            root=root,
            languages=set(data.get("languages", [])),
            frameworks=set(data.get("frameworks", [])),
            key_files=data.get("key_files", []),
            directories=data.get("directories", []),
            central_files=data.get("central_files", []),
            semantic_summaries=data.get("semantic_summaries", {}),
            project_understanding=data.get("project_understanding", ""),
            project_understanding_structured=data.get("project_understanding_structured", {}),
            files_modified=data.get("files_modified", []),
            modification_summaries=data.get("modification_summaries", {}),
            decisions=data.get("decisions", []),
            hypotheses=data.get("hypotheses", []),
            files_created=data.get("files_created", []),
            plan=data.get("plan", []),
            evaluation=Evaluation(
                status=data.get("evaluation", {}).get("status", "UNKNOWN"),
                evidence=data.get("evaluation", {}).get("evidence", []),
                rationale=data.get("evaluation", {}).get("rationale", ""),
            ),
        )

    def save(self) -> None:
        payload = {
            "languages": sorted(self.languages),
            "frameworks": sorted(self.frameworks),
            "key_files": self.key_files,
            "directories": self.directories,
            "central_files": self.central_files,
            "semantic_summaries": self.semantic_summaries,
            "project_understanding": self.project_understanding,
            "project_understanding_structured": self.project_understanding_structured,
            "files_modified": self.files_modified,
            "modification_summaries": self.modification_summaries,
            "decisions": self.decisions[-50:],
            "hypotheses": self.hypotheses[-50:],
            "files_created": self.files_created[-100:],
            "plan": self.plan,
            "evaluation": {
                "status": self.evaluation.status,
                "evidence": self.evaluation.evidence[-20:],
                "rationale": self.evaluation.rationale,
            },
        }
        path = self.root / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def update_from_report(self, report) -> None:
        self.languages = set(report.languages)
        self.frameworks = set(report.frameworks)
        self.key_files = report.key_files if hasattr(report, "key_files") else self.key_files
        self.directories = self._snapshot_directories(self.root)
        if hasattr(report, "central_files"):
            self.central_files = report.central_files
        # semantic_summaries are accumulated during inspection

    def add_decision(self, text: str) -> None:
        self.decisions.append(text)

    def add_hypothesis(self, text: str) -> None:
        self.hypotheses.append(text)

    def record_file(self, path: str) -> None:
        if path not in self.files_created:
            self.files_created.append(path)

    def summary(self) -> str:
        parts = ["Estado del proyecto:"]
        if self.languages:
            parts.append(f"Lenguajes: {', '.join(sorted(self.languages))}")
        if self.frameworks:
            parts.append(f"Frameworks: {', '.join(sorted(self.frameworks))}")
        if self.key_files:
            parts.append("Archivos clave:")
            parts.extend([f"- {k}" for k in self.key_files[:10]])
        if self.central_files:
            parts.append("Archivos centrales (heurístico):")
            parts.extend([f"- {c}" for c in self.central_files[:5]])
        if self.semantic_summaries:
            parts.append("Resúmenes semánticos (disponibles):")
            for name, summary in list(self.semantic_summaries.items())[:3]:
                parts.append(f"- {name}: {summary[:120]}...")
        if self.project_understanding:
            parts.append("Síntesis semántica (repo):")
            parts.append(f"- {self.project_understanding[:140]}...")
        if self.project_understanding_structured:
            b = self.project_understanding_structured.get("behavior")
            p = self.project_understanding_structured.get("purpose")
            if b or p:
                parts.append("Síntesis estructurada (repo):")
                if b:
                    parts.append(f"- Comportamiento: {b[:140]}...")
                if p:
                    parts.append(f"- Propósito: {p[:140]}...")
        if self.files_modified:
            parts.append("Archivos modificados:")
            parts.extend([f"- {path}" for path in self.files_modified[-5:]])
        if self.directories:
            parts.append("Estructura (niveles superiores):")
            parts.extend([f"- {d}" for d in self.directories[:15]])
        if self.hypotheses:
            parts.append("Hipótesis:")
            parts.extend([f"- {h}" for h in self.hypotheses[-5:]])
        if self.plan:
            parts.append("Plan:")
            parts.extend([f"- {p}" for p in self.plan])
        if self.decisions:
            parts.append("Decisiones recientes:")
            parts.extend([f"- {d}" for d in self.decisions[-5:]])
        if self.evaluation and self.evaluation.status != "UNKNOWN":
            parts.append(f"Evaluación: {self.evaluation.status}")
            if self.evaluation.rationale:
                parts.append(f"- {self.evaluation.rationale}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _snapshot_directories(self, root: Path, max_depth: int = 2) -> List[str]:
        entries: List[str] = []
        for path in root.glob("**/*"):
            if path.is_dir() and len(path.relative_to(root).parts) <= max_depth:
                entries.append(str(path.relative_to(root)))
            if len(entries) >= 100:
                break
        return entries
