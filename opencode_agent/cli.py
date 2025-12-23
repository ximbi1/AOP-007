from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from cmd import Cmd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from . import agent, analyzer, backend, git_tools, planner, tools


@dataclass
class SessionState:
    root: Path
    confirm: tools.ConfirmConfig
    plan: planner.Plan = field(default_factory=planner.Plan)
    last_analysis: analyzer.WorkspaceReport | None = None
    backend_manager: backend.BackendManager = field(default_factory=backend.BackendManager)

    def resolve(self, path: str | None) -> str:
        if not path:
            return str(self.root)
        return str((self.root / path).resolve())


class AssistantShell(Cmd):
    intro = "Cortex Agent Shell. Type help or ? to list commands."
    prompt = "cortex> "

    def __init__(self, state: SessionState):
        super().__init__()
        self.state = state

    # Navigation helpers
    def do_cd(self, arg: str) -> None:
        """cd <path>: change working root for the session."""
        target = Path(self.state.resolve(arg))
        if target.is_dir():
            self.state.root = target
            print(f"Root set to {target}")
        else:
            print("Not a directory")

    def do_pwd(self, arg: str) -> None:  # noqa: ARG002
        """Show current root."""
        print(self.state.root)

    # Analysis
    def do_analyze(self, arg: str) -> None:
        """analyze [path]: scan workspace and summarize languages/frameworks."""
        target = arg.strip() or str(self.state.root)
        report = analyzer.analyze_workspace(target)
        self.state.last_analysis = report
        print(report.to_summary())

    def do_agent(self, arg: str) -> None:
        """agent <objetivo> [max_attempts]: run iterative agent loop."""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: agent <objetivo> [max_attempts]")
            return
        objective = parts[0]
        max_attempts = int(parts[1]) if len(parts) > 1 else 3
        sink = ConsoleEventSink()
        runner = agent.IterativeAgent(
            root=self.state.root,
            backend_manager=self.state.backend_manager,
            confirm_cfg=self.state.confirm,
            event_sink=sink,
        )
        try:
            state = runner.run(objective, max_attempts=max_attempts)
            if state.evaluation.status == "SUCCESS":
                print(_render_success_summary(state, objective, self.state.backend_manager))
            else:
                print(state.summary())
        except Exception as exc:  # noqa: BLE001
            print(f"agent run failed: {exc}")

    # Planning
    def do_plan(self, arg: str) -> None:
        """plan add <text> | plan show | plan clear | plan complete <idx> | plan start <idx>."""
        parts = shlex.split(arg)
        if not parts:
            print(self.state.plan.to_text())
            return
        action = parts[0]
        if action == "add" and len(parts) > 1:
            self.state.plan.add(" ".join(parts[1:]))
            print(self.state.plan.to_text())
        elif action == "show":
            print(self.state.plan.to_text())
        elif action == "clear":
            self.state.plan.clear()
            print("Plan cleared")
        elif action in {"complete", "start", "block"} and len(parts) == 2:
            try:
                idx = int(parts[1]) - 1
                if action == "complete":
                    self.state.plan.complete(idx)
                elif action == "start":
                    self.state.plan.start(idx)
                else:
                    self.state.plan.block(idx)
                print(self.state.plan.to_text())
            except Exception as exc:  # noqa: BLE001
                print(f"Invalid plan operation: {exc}")
        else:
            print("Usage: plan add <text> | show | clear | complete <n> | start <n> | block <n>")

    # File operations
    def do_list(self, arg: str) -> None:
        """list [path]: list files up to depth 2."""
        target = arg.strip() or str(self.state.root)
        files = tools.list_files(target)
        for item in files:
            print(item)
        if not files:
            print("(no files)")

    def do_read(self, arg: str) -> None:
        """read <path> [start] [lines]: show file snippet."""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: read <path> [start] [lines]")
            return
        path = self.state.resolve(parts[0])
        start = int(parts[1]) if len(parts) > 1 else 1
        lines = int(parts[2]) if len(parts) > 2 else 200
        try:
            output = tools.read_file(path, start=start, lines=lines)
            print(output)
        except Exception as exc:  # noqa: BLE001
            print(f"read failed: {exc}")

    def do_search(self, arg: str) -> None:
        """search <regex> [path]: regex search across files."""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: search <regex> [path]")
            return
        pattern = parts[0]
        target = self.state.resolve(parts[1]) if len(parts) > 1 else str(self.state.root)
        results = tools.search(pattern, target)
        for line in results:
            print(line)
        if not results:
            print("(no matches)")

    def do_write(self, arg: str) -> None:
        """write <path>: prompt for new content, show diff, and apply with confirmation."""
        parts = shlex.split(arg)
        if not parts:
            print("Usage: write <path>")
            return
        target = self.state.resolve(parts[0])
        print("Enter new content. Finish with a single line containing EOF.")
        lines: List[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "EOF":
                break
            lines.append(line)
        content = "\n".join(lines) + ("\n" if lines else "")
        try:
            result = tools.safe_write(target, content, self.state.confirm)
            print(result)
        except Exception as exc:  # noqa: BLE001
            print(f"write failed: {exc}")

    # Commands
    def do_run(self, arg: str) -> None:
        """run <command>: execute a shell command with confirmation."""
        if not arg.strip():
            print("Usage: run <command>")
            return
        cmd = shlex.split(arg)
        try:
            result = tools.run_command(cmd, self.state.confirm, cwd=str(self.state.root))
            print(tools.print_command_result(result))
        except Exception as exc:  # noqa: BLE001
            print(f"command cancelled or failed: {exc}")

    # Git helpers
    def do_git_status(self, arg: str) -> None:  # noqa: ARG002
        """Show git status."""
        print(git_tools.status(str(self.state.root)))

    def do_git_diff(self, arg: str) -> None:  # noqa: ARG002
        """Show git diff."""
        print(git_tools.diff(str(self.state.root)))

    def do_git_commit(self, arg: str) -> None:
        """git_commit <message>: create a commit with automatic add."""
        if not arg.strip():
            print("Usage: git_commit <message>")
            return
        message = arg.strip()
        print(git_tools.commit(message, self.state.confirm, str(self.state.root)))

    def do_git_revert(self, arg: str) -> None:  # noqa: ARG002
        """Revert last commit (hard reset)."""
        print(git_tools.revert_last(self.state.confirm, str(self.state.root)))

    # Session
    def do_state(self, arg: str) -> None:  # noqa: ARG002
        """Show last analysis and current plan."""
        if self.state.last_analysis:
            print(self.state.last_analysis.to_summary())
        else:
            print("No analysis cached yet")
        print(self.state.plan.to_text())

    # Exit shortcuts
    def do_exit(self, arg: str) -> bool:  # noqa: ARG002
        return True

    def do_quit(self, arg: str) -> bool:  # noqa: ARG002
        return True

    def emptyline(self) -> bool:  # type: ignore[override]
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cortex Agent Shell CLI")
    parser.add_argument("--yes", action="store_true", help="assume yes for confirmations")
    sub = parser.add_subparsers(dest="command")

    interactive = sub.add_parser("interactive", help="launch interactive shell")
    interactive.add_argument("path", nargs="?", default=".")

    analyze_cmd = sub.add_parser("analyze", help="analyze workspace")
    analyze_cmd.add_argument("path", nargs="?", default=".")

    plan_cmd = sub.add_parser("plan", help="create a quick plan")
    plan_cmd.add_argument("steps", nargs="*", help="Plan steps; each argument is one step")

    agent_cmd = sub.add_parser("agent", help="run iterative agent loop")
    agent_cmd.add_argument("objective", help="Goal for the agent")
    agent_cmd.add_argument("--max-attempts", type=int, default=3)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    confirm_cfg = tools.ConfirmConfig(assume_yes=args.yes)
    state = SessionState(root=Path(os.getcwd()), confirm=confirm_cfg)

    if args.command == "interactive":
        state.root = Path(args.path).resolve()
        with OutputLogger(state.root / "outputlog.txt"):
            AssistantShell(state).cmdloop()
        return 0
    if args.command == "analyze":
        report = analyzer.analyze_workspace(args.path)
        print(report.to_summary())
        return 0
    if args.command == "plan":
        for step in args.steps:
            state.plan.add(step)
        print(state.plan.to_text())
        return 0
    if args.command == "agent":
        sink = ConsoleEventSink()
        with OutputLogger(state.root / "outputlog.txt"):
            runner = agent.IterativeAgent(
                root=state.root,
                backend_manager=state.backend_manager,
                confirm_cfg=confirm_cfg,
                event_sink=sink,
            )
            result = runner.run(args.objective, max_attempts=args.max_attempts)
            if result.evaluation.status == "SUCCESS":
                print(_render_success_summary(result, args.objective, state.backend_manager))
            else:
                print(result.summary())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())


class ConsoleEventSink:
    def emit(self, event: str, message: str) -> None:
        labels = {
            "phase_changed": ("[phase]", "\033[34m"),
            "intention_set": ("[intent]", "\033[36m"),
            "command_proposed": ("[action]", "\033[33m"),
            "command_cancelled": ("[cancelled]", "\033[31m"),
            "strategy_changed": ("[strategy]", "\033[35m"),
            "stopping_recommended": ("[stop]", "\033[35m"),
            "success_detected": ("[success]", "\033[32m"),
        }
        label, color = labels.get(event, ("[event]", ""))
        if _use_color():
            print(f"{color}{label}\033[0m {message}")
        else:
            print(f"{label} {message}")


class OutputLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _Tee(self._stdout, self._file)
        sys.stderr = _Tee(self._stderr, self._file)
        self._file.write("\n=== Cortex Agent Session Start ===\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file:
            self._file.write("=== Cortex Agent Session End ===\n")
            self._file.flush()
        if self._stdout:
            sys.stdout = self._stdout
        if self._stderr:
            sys.stderr = self._stderr
        if self._file:
            self._file.close()


class _Tee:
    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)

    def flush(self):
        self.primary.flush()
        self.secondary.flush()

    def isatty(self):
        return bool(getattr(self.primary, "isatty", lambda: False)())


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _dedupe_paths(paths: List[str]) -> List[str]:
    seen = {}
    for path in paths:
        key = path.lower()
        if key not in seen:
            seen[key] = path
    return list(seen.values())


def _summary_is_valid(summary: str, inspected: List[str], created: List[str], modified: List[str]) -> bool:
    lower = summary.lower()
    if "{" in summary or "}" in summary:
        return False
    if "modified" in lower and not modified:
        return False
    if "created" in lower and not created:
        return False
    if "inspected" in lower and not inspected:
        return False
    return True


def _fallback_success_summary(
    objective: str,
    inspected: List[str],
    created: List[str],
    modified: List[str],
    understanding: dict,
) -> str:
    lines = ["Done.", "", "Key actions"]
    if inspected:
        for name in inspected:
            lines.append(f"- Inspected {Path(name).name}.")
    if created:
        for name in created:
            lines.append(f"- Created {Path(name).name}.")
    if modified:
        for name in modified:
            lines.append(f"- Modified {Path(name).name}.")
    if not (inspected or created or modified):
        lines.append("- No observable actions were recorded.")

    lines.append("")
    lines.append("Result")
    if created:
        for name in created:
            lines.append(f"- File {Path(name).name} exists in the workspace.")
    if modified and not created:
        lines.append("- Updated files are now present in the workspace.")
    lines.append("- No runtime behavior was executed or modified.")
    lines.append("")
    lines.append("Constraints")
    lines.append("- No execution performed.")
    lines.append("- Based on inspected content only.")
    return "\n".join(lines)


def _render_success_summary(state: agent.ProjectState, objective: str, backend_manager: backend.BackendManager) -> str:
    return _render_success_summary_strict(state, objective, backend_manager)


def _render_success_summary_strict(
    state: agent.ProjectState,
    objective: str,
    backend_manager: backend.BackendManager,
) -> str:
    inspected = list(state.semantic_summaries.keys())[:5]
    created = _dedupe_paths(state.files_created[-10:])
    modified = _dedupe_paths(state.files_modified[-10:])
    created_lower = {path.lower() for path in created}
    modified = [path for path in modified if path.lower() not in created_lower]

    lines = ["Done.", "", "Key actions"]
    if inspected:
        for name in inspected:
            lines.append(f"- Inspected {Path(name).name}.")
    if created:
        for name in created:
            lines.append(f"- Created {Path(name).name}.")
    if modified:
        for name in modified:
            lines.append(f"- Modified {Path(name).name}.")
    if not (inspected or created or modified):
        lines.append("- No observable actions were recorded.")

    lines.append("")
    lines.append("Result")
    result_lines = _build_result_lines(state, created, modified)
    if result_lines:
        lines.extend([f"- {line}" for line in result_lines])
    else:
        lines.append("- No observable file changes were recorded.")
    explanation = _maybe_generate_explanation(state, created, modified, backend_manager)
    if explanation:
        lines.append("")
        lines.append("Explanation")
        lines.append(f"- {explanation}")
    return "\n".join(lines)


def _build_result_lines(state: agent.ProjectState, created: List[str], modified: List[str]) -> List[str]:
    result_lines: List[str] = []
    code_target = _select_code_target(created, state.create_primary_target)
    readme_created = any(Path(path).name.lower() == "readme.md" for path in created)

    if state.create_requires_code and code_target:
        language = _language_from_extension(Path(code_target).suffix)
        descriptor = f"{language} code file" if language else "Code file"
        result_lines.append(f"{descriptor} `{Path(code_target).name}` is now present.")
    elif created:
        for path in created:
            result_lines.append(f"File `{Path(path).name}` is now present.")

    if readme_created and (state.create_doc_requested or not state.create_requires_code):
        result_lines.append("Documentation file `README.md` is now present.")

    if modified and not created:
        result_lines.append("Updated files are now present with the latest changes.")
    return result_lines


def _select_code_target(created: List[str], primary_target: str | None) -> str | None:
    if primary_target:
        target_name = Path(primary_target).name.lower()
        for path in created:
            if Path(path).name.lower() == target_name:
                return path
    for path in created:
        if _language_from_extension(Path(path).suffix):
            return path
    return None


def _language_from_extension(ext: str) -> str | None:
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".sh": "Shell",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".java": "Java",
        ".cs": "C#",
    }
    return mapping.get(ext.lower())


def _maybe_generate_explanation(
    state: agent.ProjectState,
    created: List[str],
    modified: List[str],
    backend_manager: backend.BackendManager,
) -> str | None:
    semantic_payload = _build_explanation_payload(state, created, modified)
    if not semantic_payload:
        return None
    allowed_files = semantic_payload.get("allowed_files", [])
    try:
        with backend_manager:
            response = backend.call_llama(
                backend_manager.settings.base_url,
                {
                    "model": "local-model",
                    "messages": _build_explanation_messages(semantic_payload),
                    "temperature": 0.2,
                },
            )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None
    explanation = _sanitize_explanation(content)
    return _validate_explanation(explanation, allowed_files, semantic_payload.get("readme_created", False))


def _build_explanation_payload(
    state: agent.ProjectState,
    created: List[str],
    modified: List[str],
) -> dict | None:
    if not (created or modified):
        return None
    summaries = {
        name: summary
        for name, summary in state.semantic_summaries.items()
        if name in {Path(path).name for path in created + modified}
    }
    if not summaries and not state.project_understanding_structured:
        return None
    code_target = _select_code_target(created, state.create_primary_target)
    return {
        "created_files": [Path(path).name for path in created],
        "modified_files": [Path(path).name for path in modified],
        "primary_code_file": Path(code_target).name if code_target else None,
        "language": _language_from_extension(Path(code_target).suffix) if code_target else None,
        "semantic_summaries": summaries,
        "project_understanding": state.project_understanding_structured,
        "allowed_files": sorted({Path(path).name for path in created + modified}),
        "readme_created": any(Path(path).name.lower() == "readme.md" for path in created),
    }


def _build_explanation_messages(payload: dict) -> List[dict]:
    system = (
        "Eres un narrador técnico en español. Explica el resultado en 1-2 frases claras y humanas. "
        "Usa solo hechos explícitos del payload. "
        "No inventes archivos ni funciones. No uses lenguaje especulativo. "
        "No menciones archivos fuera de allowed_files."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _sanitize_explanation(content: str) -> str:
    if not content:
        return ""
    text = content.strip()
    if text.startswith("-"):
        text = text.lstrip("- ")
    return " ".join(text.split())


def _validate_explanation(text: str, allowed_files: List[str], readme_created: bool) -> str | None:
    if not text:
        return None
    lower = text.lower()
    forbidden = [
        "probablemente",
        "podria",
        "podría",
        "quizá",
        "quizas",
        "maybe",
        "perhaps",
        "might",
        "may ",
        "could",
        "would",
        "posiblemente",
    ]
    if any(token in lower for token in forbidden):
        return None
    if "readme" in lower and not readme_created:
        return None
    if ("documentation" in lower or "documentación" in lower) and not readme_created:
        return None
    allowed_lower = {name.lower() for name in allowed_files}
    mentioned = {
        match.lower()
        for match in re.findall(r"[\w./-]+\.(?:py|js|ts|tsx|sh|go|rs|rb|php|java|cs|md|txt)", text)
    }
    if any(name not in allowed_lower for name in mentioned):
        return None
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) > 2:
        return None
    return text
