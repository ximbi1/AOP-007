from pathlib import Path
from types import SimpleNamespace

from opencode_agent import cli, state


class DummyBackend:
    def __init__(self):
        self.settings = SimpleNamespace(base_url="http://fake")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_success_summary_includes_valid_explanation(monkeypatch):
    def fake_call_llama(base_url, payload, timeout=0):  # noqa: ARG001
        return {"choices": [{"message": {"content": "The script implements a basic calculator."}}]}

    monkeypatch.setattr(cli.backend, "call_llama", fake_call_llama)

    project_state = state.ProjectState(root=Path("."))
    project_state.files_created = ["script.py"]
    project_state.semantic_summaries = {"script.py": "Imprime información en la consola."}
    project_state.create_requires_code = True
    project_state.create_primary_target = "script.py"

    summary = cli._render_success_summary(project_state, "objetivo", DummyBackend())

    assert "Explanation" in summary
    assert "The script implements a basic calculator." in summary


def test_success_summary_drops_invalid_explanation(monkeypatch):
    def fake_call_llama(base_url, payload, timeout=0):  # noqa: ARG001
        return {"choices": [{"message": {"content": "Creates README.md and more."}}]}

    monkeypatch.setattr(cli.backend, "call_llama", fake_call_llama)

    project_state = state.ProjectState(root=Path("."))
    project_state.files_created = ["script.py"]
    project_state.semantic_summaries = {"script.py": "Imprime información en la consola."}
    project_state.create_requires_code = True
    project_state.create_primary_target = "script.py"

    summary = cli._render_success_summary(project_state, "objetivo", DummyBackend())

    assert "Explanation" not in summary
