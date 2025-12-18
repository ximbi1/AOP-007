from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import analyzer, backend, tools
from .understanding import ProjectUnderstanding
from .state import Evaluation, ProjectState


@dataclass
class Attempt:
    action: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    evaluation: str = ""
    correction: str = ""
    status: str = "PARTIAL"  # SUCCESS|PARTIAL|FAILURE
    evidence: List[str] = field(default_factory=list)


@dataclass
class Action:
    kind: str  # run|write|none
    description: str
    command: List[str] | None = None
    path: Optional[str] = None
    content: Optional[str] = None

    @classmethod
    def from_model(cls, data: Dict[str, object]) -> "Action":
        kind = str(data.get("type") or data.get("kind") or "none").lower()
        description = str(data.get("reason") or data.get("description") or "")
        path = data.get("path")
        content = data.get("content")
        command_val = data.get("command")
        if isinstance(command_val, str):
            command = shlex.split(command_val)
        elif isinstance(command_val, list):
            command = [str(x) for x in command_val]
        else:
            command = None
        return cls(kind=kind, description=description, command=command, path=path if isinstance(path, str) else None, content=content if isinstance(content, str) else None)


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:  # noqa: BLE001
                        break
        start = text.find("{", start + 1)
    return None


class IterativeAgent:
    def __init__(
        self,
        root: Path,
        backend_manager: backend.BackendManager,
        confirm_cfg: tools.ConfirmConfig,
        model: str = "local-model",
        event_sink: Optional[object] = None,
    ) -> None:
        self.root = root
        self.backend = backend_manager
        self.confirm_cfg = confirm_cfg
        self.model = model
        self.state = ProjectState.load(root)
        self.messages: List[Dict[str, object]] = []
        self.current_objective: str = ""
        self.event_sink = event_sink or _NoOpEvents()
        self.objective_type: str = "UNKNOWN"

    # ------------------------------------------------------------------
    def run(self, objective: str, max_attempts: int = 3) -> ProjectState:
        self._emit("phase_changed", "Analizando…")
        self.current_objective = objective
        self.objective_type = self._infer_objective_type(objective)
        analysis = analyzer.analyze_workspace(str(self.root))
        self.state.update_from_report(analysis)
        self.state.add_decision(f"Objetivo: {objective}")
        self.state.add_hypothesis("Comenzar con inspección y plan del modelo")
        self.state.save()

        understanding = ProjectUnderstanding(self.root).build()
        self.state.add_hypothesis("Pistas semánticas disponibles (hipótesis, requieren confirmación)")
        self.semantic_summary = understanding.to_summary()

        with self.backend:
            self._bootstrap_messages(objective)
            self._emit("phase_changed", "Planificando…")
            self.state.plan = self._request_plan()
            self.state.save()
            for attempt_no in range(1, max_attempts + 1):
                self._emit("phase_changed", "Ejecutando…")
                action = self._request_action(attempt_no)
                self._emit("intention_set", f"Intento {attempt_no}: {action.description or action.kind}")
                result = self._execute_action(action)
                self._emit("phase_changed", "Evaluando…")
                evaluation = self._evaluate_goal(objective, result)
                result.status = evaluation.status
                result.evidence.extend(evaluation.evidence)
                self.state.evaluation = evaluation
                self.state.add_decision(f"Intento {attempt_no}: {result.evaluation}")
                self.state.save()
                if evaluation.status == "SUCCESS":
                    self.state.add_decision(
                        "No haré nada porque el objetivo ya está cumplido"
                    )
                    self._emit("success_detected", "Evidencia suficiente de cumplimiento del objetivo")
                    break
                if attempt_no >= max_attempts:
                    self.state.add_decision(
                        "He intentado varias veces y recomiendo parar"
                    )
                    self._emit("stopping_recommended", "Límite de intentos alcanzado")
                    break
                strategy, justification, discarded = self._choose_strategy(
                    evaluation, attempt_no, max_attempts
                )
                self.state.add_decision(
                    f"Estrategia: {strategy} (descarto: {', '.join(discarded)}) porque {justification}"
                )
                self._emit("strategy_changed", f"{strategy} :: {justification}")
                if strategy in {"ask_user", "stop"}:
                    break
                correction = self._request_correction(result, strategy, discarded)
                result.correction = correction or "sin corrección"
                if self.state.evaluation.status == "SUCCESS":
                    break
        return self.state

    # ------------------------------------------------------------------
    def _bootstrap_messages(self, objective: str) -> None:
        system_prompt = (
            "Eres un agente de desarrollo en terminal. Trabajas en bucles de "
            "objetivo → plan → acción → evaluación → corrección. Responde en español. "
            "Siempre devuelve JSON en tus mensajes de herramienta: para planes usa {\"plan\": [...], \"hypotheses\": [...]}. "
            "Para acciones usa {\"action\": {\"type\": \"run|write|none\", \"command\": str, \"path\": str, \"content\": str, \"reason\": str}}."
        )
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Objetivo: {objective}\n{self.state.summary()}\n{getattr(self, 'semantic_summary', '')}",
            },
        ]

    def _request_plan(self) -> List[str]:
        prompt = (
            "Genera un plan breve en viñetas para cumplir el objetivo. "
            "Responde solo JSON con claves plan (lista) e hypotheses (lista)."
        )
        messages = self.messages + [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)
        data = _extract_json(response) or {}
        plan = data.get("plan") if isinstance(data, dict) else []
        hypotheses = data.get("hypotheses") if isinstance(data, dict) else []
        if isinstance(plan, list):
            self.state.plan = [str(p) for p in plan]
        if isinstance(hypotheses, list):
            for h in hypotheses:
                self.state.add_hypothesis(str(h))
        if not self.state.plan:
            self.state.plan = ["Analizar entorno", "Probar acción mínima", "Evaluar resultados"]
        self.messages.append({"role": "assistant", "content": response})
        return self.state.plan

    def _request_action(self, attempt_no: int) -> Action:
        user_prompt = (
            "Sigue el ciclo: presenta la siguiente acción concreta. "
            "Responde JSON con la clave action. Prioriza comandos pequeños y lecturas antes de escribir."
        )
        messages = self.messages + [
            {
                "role": "user",
                "content": f"Intento {attempt_no}. Estado:\n{self.state.summary()}\n{user_prompt}",
            }
        ]
        response = self._call_llm(messages)
        self.messages.append({"role": "assistant", "content": response})
        data = _extract_json(response) or {}
        raw_action = data.get("action") if isinstance(data, dict) else None
        if isinstance(raw_action, dict):
            return Action.from_model(raw_action)
        return Action(kind="none", description="Sin acción propuesta")

    def _request_correction(self, result: Attempt, strategy: str, discarded: List[str]) -> str:
        prompt = (
            "La última acción no demuestra el objetivo. Estrategia elegida: {strategy}. "
            "Descartadas: {discarded}. Devuelve JSON con action y correction que explique la hipótesis.".format(
                strategy=strategy, discarded=", ".join(discarded)
            )
        )
        messages = self.messages + [
            {
                "role": "user",
                "content": (
                    f"Evaluación actual: {self.state.evaluation.status}\n"
                    f"Evidencia: {self.state.evaluation.evidence}\n"
                    f"Razonamiento: {self.state.evaluation.rationale}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}\n{prompt}"
                ),
            }
        ]
        response = self._call_llm(messages)
        self.messages.append({"role": "assistant", "content": response})
        data = _extract_json(response) or {}
        corr = ""
        if isinstance(data, dict):
            if corr_val := data.get("correction"):
                corr = str(corr_val)
            if action := data.get("action"):
                act = Action.from_model(action)
                next_result = self._execute_action(act)
                evaluation = self._evaluate_goal(self.current_objective, next_result)
                next_result.status = evaluation.status
                self.state.evaluation = evaluation
                if next_result.status == "SUCCESS":
                    self.state.add_decision(f"Corrección exitosa: {act.description}")
                else:
                    self.state.add_decision("Corrección fallida")
                self.state.save()
        return corr

    def _call_llm(self, messages: List[Dict[str, object]]) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": messages,
        }
        response = backend.call_llama(self.backend.settings.base_url, payload, timeout=180)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or "{}"

    # ------------------------------------------------------------------
    def _execute_action(self, action: Action) -> Attempt:
        if action.kind == "run" and action.command:
            try:
                self._emit("command_proposed", f"Ejecutar: {' '.join(action.command)}")
                result = tools.run_command(action.command, self.confirm_cfg, cwd=str(self.root))
            except Exception as exc:  # noqa: BLE001
                self._emit("command_cancelled", str(exc))
                return Attempt(
                    action="run",
                    returncode=None,
                    evaluation=str(exc),
                    status="FAILURE",
                )
            evaluation, status, evidence = self._evaluate_process(
                result.stdout, result.stderr, result.returncode, action.command
            )
            return Attempt(
                action=" ".join(action.command),
                returncode=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                evaluation=evaluation,
                status=status,
                evidence=evidence,
            )
        if action.kind == "write" and action.path and action.content is not None:
            try:
                self._emit("command_proposed", f"Escribir: {action.path}")
                message = tools.safe_write(str(Path(self.root, action.path)), action.content, self.confirm_cfg)
                self.state.record_file(action.path)
                evaluation = f"write -> {message}"
                return Attempt(
                    action=f"write {action.path}",
                    evaluation=evaluation,
                    status="PARTIAL",
                    evidence=[evaluation],
                )
            except Exception as exc:  # noqa: BLE001
                return Attempt(
                    action=f"write {action.path}",
                    evaluation=str(exc),
                    status="FAILURE",
                )
        return Attempt(
            action=action.description or action.kind,
            evaluation="sin acción ejecutada",
            status="PARTIAL",
            evidence=["No se ejecutó ninguna acción"],
        )

    def _evaluate_process(
        self, stdout: str, stderr: str, returncode: int, command: List[str]
    ) -> tuple[str, str, List[str]]:
        signals = []
        evidence = []
        if returncode == 0:
            evidence.append("Comando terminó con código 0")
        else:
            signals.append(f"returncode={returncode}")
        combined = f"{stdout}\n{stderr}".lower()
        for marker in ("error", "exception", "traceback", "failed", "assert"):
            if marker in combined:
                signals.append(marker)
        status = "FAILURE" if signals else "SUCCESS"
        evaluation = "; ".join(signals) if signals else "ejecución exitosa"
        if "test" in " ".join(command).lower() and returncode == 0:
            evidence.append("Tests pasaron")
        if stdout:
            evidence.append(f"stdout observado ({min(len(stdout),80)} chars)")
        if stderr:
            evidence.append(f"stderr observado ({min(len(stderr),80)} chars)")
            evidence.append(f"stderr sample: {stderr[:80].strip()}")
        return evaluation, status, evidence

    def _evaluate_goal(self, objective: str, attempt: Attempt) -> Evaluation:
        evidence = list(attempt.evidence)
        status = attempt.status
        # CREATE_ARTIFACT handling: success if a file was written/created and no failure observed
        if self.objective_type == "CREATE_ARTIFACT" and status != "FAILURE":
            artifact_signal = False
            if self.state.files_created:
                artifact_signal = True
                evidence.append(f"Archivos creados: {', '.join(self.state.files_created[-3:])}")
            if attempt.action.startswith("write"):
                artifact_signal = True
                evidence.append("Acción de escritura realizada")
            if artifact_signal:
                status = "SUCCESS"
                evidence.append("Objetivo de creación cumplido (artefacto generado)")
        if attempt.status == "SUCCESS":
            if "test" in attempt.action.lower():
                status = "SUCCESS"
                evidence.append("Tests relacionados con el objetivo pasaron")
            else:
                status = "PARTIAL"
        if attempt.status == "PARTIAL" and self.state.files_created:
            evidence.append(f"Archivos creados: {', '.join(self.state.files_created[-3:])}")
        if attempt.returncode not in (None, 0):
            evidence.append(f"Código de salida {attempt.returncode}")

        rationale = (
            "Evidencia observada: "
            + ("; ".join(evidence) if evidence else "sin evidencia sólida")
            + f". Conclusión frente al objetivo '{objective}': {status}."
        )
        return Evaluation(status=status, evidence=evidence, rationale=rationale)

    def _choose_strategy(
        self, evaluation: Evaluation, attempt_no: int, max_attempts: int
    ) -> tuple[str, str, List[str]]:
        options = ["change_approach", "simplify", "ask_user", "stop"]
        strategy = "change_approach"
        justification = evaluation.rationale or "Sin evidencia concluyente"
        lower_evidence = " ".join(evaluation.evidence).lower()
        if any(token in lower_evidence for token in ("ambig", "unclear", "duda")):
            strategy = "ask_user"
            discarded = [opt for opt in options if opt != strategy]
            return strategy, "Instrucción ambigua; pedir decisión humana", discarded
        if evaluation.status == "FAILURE":
            strategy = "change_approach" if attempt_no < max_attempts else "ask_user"
            justification = "Fallo persistente; probar enfoque distinto" if strategy == "change_approach" else "Demasiados fallos; se requiere decisión humana"
        elif evaluation.status == "PARTIAL":
            strategy = "simplify" if attempt_no < max_attempts - 1 else "ask_user"
            justification = "Progreso parcial; simplificar pasos" if strategy == "simplify" else "Progreso insuficiente; pedir claridad"
        elif evaluation.status == "SUCCESS":
            strategy = "stop"
            justification = "Objetivo cumplido"
        discarded = [opt for opt in options if opt != strategy]
        return strategy, justification, discarded

    def _emit(self, event: str, message: str) -> None:
        try:
            if hasattr(self.event_sink, "emit"):
                self.event_sink.emit(event, message)
        except Exception:
            pass

    def _infer_objective_type(self, objective: str) -> str:
        text = objective.lower()
        create_keywords = [
            "crear",
            "crea",
            "genera",
            "generar",
            "escribe",
            "escribir",
            "script",
            "archivo",
            "fichero",
            "file",
            "generate",
            "write",
            "add file",
        ]
        if any(k in text for k in create_keywords):
            return "CREATE_ARTIFACT"
        exec_keywords = ["ejecuta", "run", "ejecutar", "execute"]
        if any(k in text for k in exec_keywords):
            return "EXECUTE"
        modify_keywords = ["modifica", "modificar", "refactor", "ajusta", "update", "cambia"]
        if any(k in text for k in modify_keywords):
            return "MODIFY"
        inspect_keywords = ["lee", "list", "inspecciona", "analiza", "consulta"]
        if any(k in text for k in inspect_keywords):
            return "INSPECT"
        return "UNKNOWN"


class _NoOpEvents:
    def emit(self, event: str, message: str) -> None:  # noqa: D401
        """No-op event sink."""
        return None
