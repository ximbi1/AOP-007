from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts", ".tsx"},
    "go": {".go"},
    "rust": {".rs"},
    "java": {".java"},
    "csharp": {".cs"},
    "cpp": {".cpp", ".cc", ".h", ".hpp"},
    "php": {".php"},
    "ruby": {".rb"},
}


FRAMEWORK_HINTS = {
    "django": ["manage.py", "settings.py"],
    "flask": ["app.py", "flask"],
    "fastapi": ["fastapi"],
    "react": ["package.json", "react"],
    "nextjs": ["next.config.js", "next.config.mjs"],
    "vue": ["vue.config.js"],
    "svelte": ["svelte.config.js", "svelte.config.ts"],
    "node": ["package.json"],
    "poetry": ["pyproject.toml"],
    "pip": ["requirements.txt"],
    "go": ["go.mod"],
    "rust": ["Cargo.toml"],
}

KEY_FILE_CANDIDATES = [
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "manage.py",
    "app.py",
    "main.py",
    "main.go",
    "index.js",
    "index.ts",
    "index.tsx",
    "Dockerfile",
    "Makefile",
    "go.mod",
    "Cargo.toml",
]


@dataclass
class WorkspaceReport:
    root: Path
    languages: Set[str] = field(default_factory=set)
    frameworks: Set[str] = field(default_factory=set)
    files: List[str] = field(default_factory=list)
    key_files: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"Root: {self.root}"]
        lines.append(f"Languages: {', '.join(sorted(self.languages)) or 'unknown'}")
        lines.append(f"Frameworks: {', '.join(sorted(self.frameworks)) or 'none detected'}")
        if self.key_files:
            lines.append("Key files:")
            lines.extend([f"- {k}" for k in self.key_files[:10]])
        if self.missing:
            lines.append("Missing scaffolding:")
            lines.extend([f"- {item}" for item in self.missing])
        if self.notes:
            lines.append("Notes:")
            lines.extend([f"- {note}" for note in self.notes])
        lines.append(f"File sample ({min(len(self.files), 20)} shown):")
        for rel in self.files[:20]:
            lines.append(f"- {rel}")
        return "\n".join(lines)


def detect_languages(files: List[Path]) -> Set[str]:
    langs: Set[str] = set()
    for path in files:
        ext = path.suffix.lower()
        for lang, exts in LANGUAGE_EXTENSIONS.items():
            if ext in exts:
                langs.add(lang)
    return langs


def detect_frameworks(files: List[Path]) -> Set[str]:
    rel_names = {file.name for file in files}
    content_hints = set()
    for file in files:
        if file.suffix in {".json", ".toml", ".js", ".ts", ".py"} and file.stat().st_size < 100_000:
            try:
                text = file.read_text(encoding="utf-8")
            except Exception:
                continue
            for fw, hints in FRAMEWORK_HINTS.items():
                for hint in hints:
                    if hint in text:
                        content_hints.add(fw)
    frameworks = set()
    for fw, hints in FRAMEWORK_HINTS.items():
        for hint in hints:
            if hint in rel_names or fw in content_hints:
                frameworks.add(fw)
    return frameworks


def detect_key_files(files: List[Path]) -> List[str]:
    keys: List[str] = []
    rel_names = {f.name: f for f in files}
    for candidate in KEY_FILE_CANDIDATES:
        if candidate in rel_names:
            keys.append(candidate)
    return keys


def find_missing_scaffolding(report: WorkspaceReport) -> List[str]:
    missing = []
    if not report.files:
        missing.append("workspace is empty; suggest initializing a project scaffold")
    if "python" in report.languages:
        if not any(file.endswith("requirements.txt") or file.endswith("pyproject.toml") for file in report.files):
            missing.append("python dependency file (requirements.txt/pyproject.toml)")
        if not any(Path(f).name.startswith("test") for f in report.files):
            missing.append("python tests (tests/ directory)")
    if "javascript" in report.languages or "typescript" in report.languages:
        if not any(file.endswith("package.json") for file in report.files):
            missing.append("package.json")
        if not any("eslint" in f.lower() for f in report.files):
            missing.append("lint configuration (.eslintrc, etc.)")
    return missing


def analyze_workspace(path: str, max_files: int = 1000) -> WorkspaceReport:
    root = Path(path).resolve()
    collected: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            collected.append(Path(dirpath, name))
            if len(collected) >= max_files:
                break
        if len(collected) >= max_files:
            break
    relative_files = [str(p.relative_to(root)) for p in collected]
    languages = detect_languages(collected)
    frameworks = detect_frameworks(collected)
    key_files = detect_key_files(collected)
    report = WorkspaceReport(
        root=root,
        languages=languages,
        frameworks=frameworks,
        files=relative_files,
        key_files=key_files,
    )
    report.missing = find_missing_scaffolding(report)
    if len(collected) >= max_files:
        report.notes.append("file scan truncated for performance")
    return report
