from pathlib import Path

from opencode_agent.state import Evaluation, ProjectState


def test_state_persistence_roundtrip(tmp_path):
    root = tmp_path
    state = ProjectState(root=root)
    state.languages = {"python"}
    state.frameworks = {"django"}
    state.key_files = ["manage.py"]
    state.directories = ["app", "tests"]
    state.plan = ["plan"]
    state.decisions.append("d1")
    state.hypotheses.append("h1")
    state.files_created.append("file.txt")
    state.evaluation = Evaluation(status="SUCCESS", evidence=["ok"], rationale="done")
    state.save()

    reloaded = ProjectState.load(root)
    assert reloaded.languages == {"python"}
    assert reloaded.frameworks == {"django"}
    assert reloaded.key_files == ["manage.py"]
    assert "d1" in reloaded.decisions
    assert "h1" in reloaded.hypotheses
    assert "file.txt" in reloaded.files_created
    assert reloaded.evaluation.status == "SUCCESS"
    assert reloaded.plan == ["plan"]


def test_state_updates_after_attempts(tmp_path):
    root = tmp_path
    state = ProjectState.load(root)
    state.decisions.append("first")
    state.evaluation = Evaluation(status="FAILURE", evidence=["rc=1"], rationale="bad")
    state.save()

    again = ProjectState.load(root)
    again.decisions.append("second")
    again.evaluation = Evaluation(status="SUCCESS", evidence=["ok"], rationale="good")
    again.save()

    final = ProjectState.load(root)
    assert "first" in final.decisions
    assert "second" in final.decisions
    assert final.evaluation.status == "SUCCESS"
    assert final.evaluation.evidence[-1] == "ok"
