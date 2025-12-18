import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencode_agent import agent, tools


def llm_stub(responses):
    calls = {"count": 0}

    def _call_llama(base_url, payload, timeout=0):  # noqa: ARG001
        try:
            content = responses[calls["count"]]
        except IndexError as exc:  # pragma: no cover - guard
            raise AssertionError("LLM called more times than expected") from exc
        calls["count"] += 1
        return {"choices": [{"message": {"content": content}}]}

    return _call_llama, calls


class DummyBackend:
    def __init__(self):
        self.settings = SimpleNamespace(base_url="http://fake")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_agent_stops_when_goal_already_met(monkeypatch, tmp_path):
    responses = [
        json.dumps({"plan": ["run tests"], "hypotheses": ["todo ok"]}),
        json.dumps({"action": {"type": "run", "command": "pytest", "reason": "verify"}}),
    ]
    call_llama, calls = llm_stub(responses)
    monkeypatch.setattr(agent.backend, "call_llama", call_llama)

    run_calls = {"count": 0}

    def fake_run(cmd, cfg, cwd=None):  # noqa: ARG001
        run_calls["count"] += 1
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(tools, "run_command", fake_run)

    ag = agent.IterativeAgent(
        root=tmp_path,
        backend_manager=DummyBackend(),
        confirm_cfg=tools.ConfirmConfig(assume_yes=True),
    )
    state = ag.run("objetivo ya cumplido", max_attempts=1)

    assert state.evaluation.status == "SUCCESS"
    assert any("objetivo ya está cumplido" in d for d in state.decisions)
    assert run_calls["count"] == 1
    assert not state.files_created
    assert calls["count"] == 2  # plan + action


def test_agent_recommends_stop_after_failures(monkeypatch, tmp_path):
    responses = [
        json.dumps({"plan": ["intento"], "hypotheses": []}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "fail"}}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "fail again"}}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "fail final"}}),
    ]
    call_llama, calls = llm_stub(responses)
    monkeypatch.setattr(agent.backend, "call_llama", call_llama)

    def fake_run(cmd, cfg, cwd=None):  # noqa: ARG001
        return SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(tools, "run_command", fake_run)

    ag = agent.IterativeAgent(
        root=tmp_path,
        backend_manager=DummyBackend(),
        confirm_cfg=tools.ConfirmConfig(assume_yes=True),
    )
    state = ag.run("objetivo difícil", max_attempts=2)

    assert state.evaluation.status in {"FAILURE", "PARTIAL"}
    assert any("recomiendo parar" in d for d in state.decisions)
    assert calls["count"] >= 3  # plan + two actions


def test_agent_asks_human_on_ambiguous(monkeypatch, tmp_path):
    responses = [
        json.dumps({"plan": ["?"], "hypotheses": []}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "ambiguous"}}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "ambiguous followup"}}),
        json.dumps({"action": {"type": "run", "command": "false", "reason": "ambiguous final"}}),
    ]
    call_llama, calls = llm_stub(responses)
    monkeypatch.setattr(agent.backend, "call_llama", call_llama)

    def fake_run(cmd, cfg, cwd=None):  # noqa: ARG001
        return SimpleNamespace(stdout="", stderr="unclear", returncode=1)

    monkeypatch.setattr(tools, "run_command", fake_run)

    ag = agent.IterativeAgent(
        root=tmp_path,
        backend_manager=DummyBackend(),
        confirm_cfg=tools.ConfirmConfig(assume_yes=True),
    )
    state = ag.run("instrucción ambigua", max_attempts=2)

    assert any("Estrategia: ask_user" in d for d in state.decisions)
    assert calls["count"] >= 2


def test_risky_action_requires_confirmation(monkeypatch, tmp_path):
    responses = [
        json.dumps({"plan": ["run"], "hypotheses": []}),
        json.dumps({"action": {"type": "run", "command": "rm -rf /", "reason": "danger"}}),
    ]
    call_llama, _ = llm_stub(responses)
    monkeypatch.setattr(agent.backend, "call_llama", call_llama)

    monkeypatch.setattr(tools, "confirm", lambda prompt, cfg: False)

    def forbidden_run(*args, **kwargs):  # pragma: no cover - ensure not hit
        raise AssertionError("Command should not execute")

    monkeypatch.setattr(tools.subprocess, "run", forbidden_run)

    ag = agent.IterativeAgent(
        root=tmp_path,
        backend_manager=DummyBackend(),
        confirm_cfg=tools.ConfirmConfig(assume_yes=False),
    )
    state = ag.run("acción riesgosa", max_attempts=1)

    assert state.evaluation.status in {"FAILURE", "PARTIAL"}
    assert any("recomiendo parar" in d or "Objetivo" in d for d in state.decisions)
    files = [p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()]
    assert files == [Path(".opencode/state.json")]
