from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


ENTRYPOINT_NAMES = {
    "main",
    "app",
    "index",
    "server",
    "cli",
}

CONFIG_NAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "Cargo.toml",
    "Makefile",
    "Dockerfile",
}

COMMON_DIRS = {"src", "lib", "app", "apps", "services", "tests"}


@dataclass
class SemanticHint:
    path: str
    reason: str
    confidence: str  # high | medium | low


@dataclass
class ProjectUnderstanding:
    root: Path
    entrypoints: List[SemanticHint] = field(default_factory=list)
    configs: List[SemanticHint] = field(default_factory=list)
    central_files: List[SemanticHint] = field(default_factory=list)
    related_modules: List[SemanticHint] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def build(self) -> "ProjectUnderstanding":
        files = list(self.root.rglob("*"))
        files = [p for p in files if p.is_file() and p.stat().st_size < 200_000]
        self._find_entrypoints(files)
        self._find_configs(files)
        self._find_related(files)
        return self

    def _find_entrypoints(self, files: List[Path]) -> None:
        for f in files:
            stem = f.stem.lower()
            if stem in ENTRYPOINT_NAMES:
                conf = "high" if f.name in {"main.py", "index.js", "main.go"} else "medium"
                self.entrypoints.append(
                    SemanticHint(
                        path=str(f.relative_to(self.root)),
                        reason=f"Nombre sugiere entrypoint ({f.name})",
                        confidence=conf,
                    )
                )

    def _find_configs(self, files: List[Path]) -> None:
        for f in files:
            if f.name in CONFIG_NAMES:
                self.configs.append(
                    SemanticHint(
                        path=str(f.relative_to(self.root)),
                        reason=f"Config conocida ({f.name})",
                        confidence="high",
                    )
                )

    def _find_related(self, files: List[Path]) -> None:
        refs: Dict[str, int] = {}
        stems: Dict[str, Path] = {}
        for f in files:
            stems[f.stem] = f
        import_re = re.compile(r"^(?:from\s+([\w\.]+)|import\s+([\w\.]+))", re.MULTILINE)
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for match in import_re.findall(text):
                name = match[0] or match[1]
                base = name.split(".")[0]
                if base in stems:
                    refs[base] = refs.get(base, 0) + 1
        for stem, count in sorted(refs.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            path = stems.get(stem)
            if not path:
                continue
            self.central_files.append(
                SemanticHint(
                    path=str(path.relative_to(self.root)),
                    reason=f"Referenciado {count} veces (import)",
                    confidence="medium" if count >= 2 else "low",
                )
            )
        for d in COMMON_DIRS:
            candidate = self.root / d
            if candidate.exists() and candidate.is_dir():
                self.related_modules.append(
                    SemanticHint(
                        path=str(candidate.relative_to(self.root)),
                        reason="Carpeta típica de código",
                        confidence="low",
                    )
                )

    def to_summary(self) -> str:
        lines: List[str] = ["Pistas semánticas (hipótesis, no garantizadas):"]
        if not any([self.entrypoints, self.configs, self.central_files, self.related_modules]):
            lines.append("(sin pistas; necesito guía humana)")
        if self.entrypoints:
            lines.append("Entrypoints probables:")
            lines.extend([f"- {h.path} ({h.confidence}) :: {h.reason}" for h in self.entrypoints])
        if self.configs:
            lines.append("Configuraciones relevantes:")
            lines.extend([f"- {h.path} ({h.confidence}) :: {h.reason}" for h in self.configs])
        if self.central_files:
            lines.append("Archivos potencialmente centrales:")
            lines.extend([f"- {h.path} ({h.confidence}) :: {h.reason}" for h in self.central_files])
        if self.related_modules:
            lines.append("Módulos/dirs relacionados:")
            lines.extend([f"- {h.path} ({h.confidence}) :: {h.reason}" for h in self.related_modules])
        lines.append("Estas son hipótesis; requieren confirmación humana.")
        return "\n".join(lines)
