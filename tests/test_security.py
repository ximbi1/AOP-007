import json
from types import SimpleNamespace

import pytest

from opencode_agent import agent, tools


def llm_stub(responses):
    def _call_llama(base_url, payload, timeout=0):  # noqa: ARG001
        content = responses.pop(0)
        return {"choices": [{"message": {"content": content}}]}

    return _call_llama


class DummyBackend:
    def __init__(self):
        self.settings = SimpleNamespace(base_url="http://fake")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_safe_write_requires_confirmation(monkeypatch, tmp_path):
    target = tmp_path / "file.txt"

    def deny(prompt, cfg):  # noqa: ARG001
        return False

    monkeypatch.setattr(tools, "confirm", deny)
    result = tools.safe_write(str(target), "content", tools.ConfirmConfig(assume_yes=False))
    assert result == "write cancelled"
    assert not target.exists()


def test_run_command_requires_confirmation(monkeypatch):
    calls = {"run": 0}

    def deny(prompt, cfg):  # noqa: ARG001
        return False

    def forbidden(*args, **kwargs):  # pragma: no cover
        calls["run"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tools, "confirm", deny)
    monkeypatch.setattr(tools.subprocess, "run", forbidden)

    with pytest.raises(RuntimeError):
        tools.run_command(["echo", "hi"], tools.ConfirmConfig(assume_yes=False))
    assert calls["run"] == 0


def test_agent_shows_diff_before_overwrite(monkeypatch, tmp_path):
    responses = [
        json.dumps({"plan": ["write"], "hypotheses": []}),
        json.dumps({"action": {"type": "write", "path": "a.txt", "content": "v1", "reason": "update"}}),
    ]
    monkeypatch.setattr(agent.backend, "call_llama", llm_stub(responses))
    monkeypatch.setattr(tools, "confirm", lambda prompt, cfg: True)

    ag = agent.IterativeAgent(
        root=tmp_path,
        backend_manager=DummyBackend(),
        confirm_cfg=tools.ConfirmConfig(assume_yes=True),
    )
    state = ag.run("escritura segura", max_attempts=1)

    assert (tmp_path / "a.txt").exists()
    assert state.files_created
