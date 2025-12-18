from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re


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

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".pytest_cache"}
IGNORED_EXTS = {".pyc", ".pyo", ".so", ".dylib", ".dll"}


@dataclass
class WorkspaceReport:
    root: Path
    languages: Set[str] = field(default_factory=set)
    frameworks: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    key_files: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    central_files: List[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"Root: {self.root}"]
        lines.append(f"Languages: {', '.join(sorted(self.languages)) or 'unknown'}")
        lines.append(f"Frameworks: {', '.join(self.frameworks) or 'none detected'}")
        if self.key_files:
            lines.append("Key files:")
            lines.extend([f"- {k}" for k in self.key_files[:10]])
        if self.missing:
            lines.append("Missing scaffolding:")
            lines.extend([f"- {item}" for item in self.missing])
        if self.notes:
            lines.append("Notes:")
            lines.extend([f"- {note}" for note in self.notes])
        if self.central_files:
            lines.append("Central files (heuristic):")
            lines.extend([f"- {c}" for c in self.central_files[:3]])
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


def detect_frameworks(files: List[Path]) -> Tuple[List[str], List[str]]:
    rel_names = {file.name: file for file in files}

    def has_file(name: str) -> bool:
        return name in rel_names

    def file_small(name: str, limit: int = 200_000) -> bool:
        p = rel_names.get(name)
        return bool(p and p.stat().st_size < limit)

    evidence: Dict[str, Dict[str, List[str]]] = {}
    weights = {"entry": 3, "config": 3, "bootstrap": 2, "structure": 1, "dep": 1}

    def add_ev(fw: str, kind: str, reason: str) -> None:
        entry = evidence.setdefault(fw, {})
        entry.setdefault(kind, []).append(reason)

    # --- Python frameworks ---
    req_texts: List[str] = []
    for name in ("requirements.txt", "pyproject.toml"):
        if file_small(name):
            try:
                req_texts.append(rel_names[name].read_text(encoding="utf-8").lower())
            except Exception:
                continue
    deps_text = "\n".join(req_texts)
    if "django" in deps_text:
        add_ev("django", "dep", "Dependencia django")
    if "fastapi" in deps_text:
        add_ev("fastapi", "dep", "Dependencia fastapi")
    if "flask" in deps_text:
        add_ev("flask", "dep", "Dependencia flask")

    if has_file("manage.py"):
        add_ev("django", "entry", "manage.py presente")
    settings = [p for p in rel_names if p.startswith("settings") and rel_names[p].suffix == ".py"]
    if settings:
        add_ev("django", "config", "settings.py presente")

    for file in files:
        if file.suffix != ".py" or file.stat().st_size > 50_000:
            continue
        try:
            text = file.read_text(encoding="utf-8").lower()
        except Exception:
            continue
        if "fastapi" in text and "fastapi" in deps_text:
            add_ev("fastapi", "bootstrap", f"import en {file.name}")
        if "flask" in text and "flask" in deps_text:
            add_ev("flask", "bootstrap", f"import en {file.name}")

    # --- Node / Frontend ---
    pkg_data = None
    pkg = rel_names.get("package.json")
    if pkg:
        try:
            pkg_data = json.loads(pkg.read_text(encoding="utf-8"))
        except Exception:
            pkg_data = None
    if pkg_data:
        deps = {
            **(pkg_data.get("dependencies") or {}),
            **(pkg_data.get("devDependencies") or {}),
        }
        deps_lower = {k.lower() for k in deps}
        if deps:
            add_ev("node", "config", "package.json con dependencias")
        for name in ("react", "vue", "svelte", "next"):
            if name in deps_lower:
                add_ev(name, "dep", f"package.json depende de {name}")
        # entrypoint scripts
        for script_name in ("start", "dev", "serve"):
            if pkg_data.get("scripts", {}).get(script_name):
                add_ev("node", "entry", f"script {script_name} definido")

    for entry in ("index.js", "index.ts", "main.js", "main.ts", "index.tsx", "main.tsx"):
        if has_file(entry):
            add_ev("node", "entry", f"{entry} presente")
            for fw in ("react", "vue", "svelte", "next"):
                add_ev(fw, "entry", f"{entry} presente")
    src_dir = any(f.parent.name == "src" for f in files)
    if src_dir:
        add_ev("node", "structure", "carpeta src presente")
        for fw in ("react", "vue", "svelte", "next"):
            add_ev(fw, "structure", "carpeta src presente")

    # --- Go / Rust ---
    if has_file("go.mod"):
        add_ev("go", "config", "go.mod presente")
    if has_file("main.go"):
        add_ev("go", "entry", "main.go presente")
    if has_file("Cargo.toml"):
        add_ev("rust", "config", "Cargo.toml presente")
    if any(f.name == "main.rs" for f in files):
        add_ev("rust", "entry", "main.rs presente")

    frameworks: List[str] = []
    hints: List[str] = []
    for fw, kinds in evidence.items():
        score = sum(weights.get(k, 1) * len(v) for k, v in kinds.items())
        reasons = [r for sub in kinds.values() for r in sub]
        if score >= 4 and ("entry" in kinds or "config" in kinds):
            frameworks.append(fw)
        else:
            hints.append(f"{fw} usado como librería (no estructural): {', '.join(reasons)}")

    return sorted(frameworks), hints


def detect_key_files(files: List[Path]) -> List[str]:
    keys: List[str] = []
    rel_names = {f.name: f for f in files}
    for candidate in KEY_FILE_CANDIDATES:
        if candidate in rel_names:
            keys.append(candidate)
    return keys


def detect_central_files(files: List[Path]) -> List[str]:
    """Heurística mínima de centralidad basada en imports entrantes."""

    # Map stem -> path
    stems: Dict[str, Path] = {f.stem: f for f in files if f.is_file()}
    incoming: Dict[str, int] = {s: 0 for s in stems}

    import_re = re.compile(r"^(?:from\s+([\w\.]+)|import\s+([\w\.]+))", re.MULTILINE)
    for f in files:
        if f.suffix not in {".py", ".js", ".ts", ".tsx"}:
            continue
        if f.stat().st_size > 100_000:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in import_re.findall(text):
            name = (match[0] or match[1]).split(".")[0]
            if name in incoming:
                incoming[name] += 1

    ranked = sorted(incoming.items(), key=lambda kv: kv[1], reverse=True)
    central: List[str] = []
    for stem, count in ranked[:3]:
        if count >= 2:  # needs at least 2 incoming refs to be considered
            path = stems.get(stem)
            if path:
                central.append(f"{path.name} (importado por {count} archivos)")
    if not central:
        central.append("No se detecta un archivo central claro (evidencia insuficiente)")
    return central


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
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            candidate = Path(dirpath, name)
            if candidate.suffix in IGNORED_EXTS:
                continue
            collected.append(candidate)
            if len(collected) >= max_files:
                break
        if len(collected) >= max_files:
            break
    relative_files = [str(p.relative_to(root)) for p in collected]
    languages = detect_languages(collected)
    frameworks, framework_hints = detect_frameworks(collected)
    key_files = detect_key_files(collected)
    central_candidates = detect_central_files(collected)
    report = WorkspaceReport(
        root=root,
        languages=languages,
        frameworks=frameworks,
        files=relative_files,
        key_files=key_files,
        central_files=central_candidates,
    )
    report.notes.extend(framework_hints)
    report.missing = find_missing_scaffolding(report)
    if len(collected) >= max_files:
        report.notes.append("file scan truncated for performance")
    return report
