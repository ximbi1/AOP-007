from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .tools import ConfirmConfig, confirm


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)


def ensure_repo(path: str, cfg: ConfirmConfig) -> str:
    root = Path(path).resolve()
    if (root / ".git").exists():
        return "git repository present"
    if not confirm(f"Initialize git repository in {root}?", cfg):
        return "git init cancelled"
    result = _run_git(["init"], root)
    return result.stdout or result.stderr or "git init done"


def status(path: str = ".") -> str:
    root = Path(path).resolve()
    result = _run_git(["status", "--short"], root)
    if result.returncode != 0:
        return result.stderr or "git status failed"
    return result.stdout or "clean"


def diff(path: str = ".") -> str:
    root = Path(path).resolve()
    result = _run_git(["diff"], root)
    if result.returncode != 0:
        return result.stderr or "git diff failed"
    return result.stdout or "(no changes)"


def commit(message: str, cfg: ConfirmConfig, path: str = ".") -> str:
    root = Path(path).resolve()
    ensure_repo(path, cfg)
    current_diff = diff(path)
    if not current_diff.strip():
        return "nothing to commit"
    print(current_diff)
    if not confirm("Create commit with the above changes?", cfg):
        return "commit cancelled"
    result = _run_git(["add", "-A"], root)
    if result.returncode != 0:
        return result.stderr
    result = _run_git(["commit", "-m", message], root)
    if result.returncode != 0:
        return result.stderr
    return result.stdout


def revert_last(cfg: ConfirmConfig, path: str = ".") -> str:
    root = Path(path).resolve()
    if not confirm("Revert last commit (git reset --hard HEAD~1)?", cfg):
        return "revert cancelled"
    result = _run_git(["reset", "--hard", "HEAD~1"], root)
    if result.returncode != 0:
        return result.stderr
    return result.stdout or "reverted"
