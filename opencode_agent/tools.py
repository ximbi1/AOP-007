from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ConfirmConfig:
    assume_yes: bool = False


def confirm(prompt: str, cfg: ConfirmConfig) -> bool:
    if cfg.assume_yes:
        return True
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def list_files(path: str, max_depth: int = 2, max_entries: int = 200) -> List[str]:
    root = Path(path).resolve()
    results: List[str] = []
    for current, dirs, files in os.walk(root):
        depth = Path(current).relative_to(root).parts
        if len(depth) > max_depth:
            dirs[:] = []
            continue
        for name in sorted(files):
            rel = Path(current, name).relative_to(root)
            results.append(str(rel))
            if len(results) >= max_entries:
                return results
    return results


def read_file(path: str, start: int = 1, lines: int = 200) -> str:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(path)
    with target.open("r", encoding="utf-8", errors="ignore") as fh:
        content = fh.readlines()
    start_index = max(start - 1, 0)
    end_index = min(start_index + lines, len(content))
    snippet = content[start_index:end_index]
    body = "".join(snippet)
    header = f"Showing lines {start}-{start + len(snippet) - 1} of {len(content)}" if snippet else "(empty file)"
    return f"{header}\n{body}"


def search(pattern: str, path: str = ".", max_results: int = 200) -> List[str]:
    compiled = re.compile(pattern)
    root = Path(path)
    matches: List[str] = []
    for file_path in root.rglob("*"):
        if file_path.is_dir() or file_path.stat().st_size > 1_000_000:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                rel = file_path.relative_to(root)
                matches.append(f"{rel}:{i}:{line}")
                if len(matches) >= max_results:
                    return matches
    return matches


def _render_diff(path: Path, new_content: str) -> str:
    old = ""
    if path.exists():
        old = path.read_text(encoding="utf-8", errors="ignore")
    diff_lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff_lines)


def safe_write(path: str, content: str, cfg: ConfirmConfig) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    diff_text = _render_diff(target, content)
    if diff_text:
        print(diff_text)
    else:
        print(f"No changes detected for {path}")
    if not confirm(f"Apply changes to {path}?", cfg):
        return "write cancelled"
    with target.open("w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Wrote {path}"


def run_command(cmd: List[str], cfg: ConfirmConfig, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    printable = " ".join(cmd)
    if not confirm(f"Run command: {printable}?", cfg):
        raise RuntimeError("command cancelled")
    return subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)


def print_command_result(result: subprocess.CompletedProcess) -> str:
    output = []
    output.append(f"exit={result.returncode}")
    if result.stdout:
        output.append("stdout:\n" + result.stdout)
    if result.stderr:
        output.append("stderr:\n" + result.stderr)
    return "\n".join(output)
