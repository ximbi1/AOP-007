from __future__ import annotations

import argparse
import json
import os
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


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _render_success_summary(state: agent.ProjectState, objective: str, backend_manager: backend.BackendManager) -> str:
    payload = {
        "objective": objective,
        "project_understanding": state.project_understanding,
        "project_understanding_structured": state.project_understanding_structured,
        "files_created": state.files_created[-5:],
        "notes": "Use human, concise language. Avoid internal terms."
    }
    system = (
        "You are a concise professional assistant. Write a short, human closing summary for a successful task. "
        "Use this structure: 'Done. <confirmation>'. Then 'Key changes' bullets and 'Result' bullets. "
        "Do not mention internal agent terms or diagnostics."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        with backend_manager:
            response = backend.call_llama(backend_manager.settings.base_url, {"model": "local-model", "messages": messages, "temperature": 0.2})
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() or "Done. Task completed successfully."
    except Exception:
        return "Done. Task completed successfully."
