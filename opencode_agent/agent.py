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
            try:
                command = shlex.split(command_val)
            except ValueError:
                command = [command_val]
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
        self.objective_scope_here: bool = False
        self.no_execute_requested: bool = False
        self.sub_objectives: List[str] = []
        self.current_sub_index: int = 0
        self._sub_files_created_base: int = 0
        self._sub_objectives_text: List[str] = []
        self._last_inspected_name: Optional[str] = None
        self._last_inspected_snippet: Optional[str] = None
        self.current_targets: List[str] = []
        self.modification_proposed: bool = False
        self.modification_approved: bool = False
        self.create_requires_code: bool = False
        self.create_primary_target: Optional[str] = None
        self.create_code_written: bool = False
        self.create_doc_requested: bool = False

    # ------------------------------------------------------------------
    def run(self, objective: str, max_attempts: int = 3) -> ProjectState:
        self._emit("phase_changed", "Analizando…")
        self.sub_objectives = self._parse_sub_objectives(objective)
        self._sub_objectives_text = list(self.sub_objectives)
        self.current_sub_index = 0
        self._set_current_subobjective(0)
        self.state.semantic_summaries.clear()
        self.state.project_understanding = ""
        self.state.project_understanding_structured = {}
        self.state.files_modified = []
        self.state.modification_summaries = {}
        analysis = analyzer.analyze_workspace(str(self.root))
        self.state.update_from_report(analysis)
        self.state.add_decision(f"Objetivo: {objective}")
        self.state.add_hypothesis("Comenzar con inspección y plan del modelo")
        self.state.save()

        understanding = ProjectUnderstanding(self.root).build()
        self.state.add_hypothesis("Pistas semánticas disponibles (hipótesis, requieren confirmación)")
        self.semantic_summary = understanding.to_summary()

        with self.backend:
            while True:
                self.state.plan = []
                self.state.evaluation = Evaluation()
                self._bootstrap_messages(self.current_objective)
                self._emit("phase_changed", "Planificando…")
                self.state.plan = self._request_plan()
                if self.objective_type == "CREATE_ARTIFACT":
                    self._ensure_semantic_synthesis()
                self.state.save()
                sub_success = False
                for attempt_no in range(1, max_attempts + 1):
                    self._emit("phase_changed", "Ejecutando…")
                    action = self._request_action(attempt_no)
                    self._emit(
                        "intention_set",
                        f"Intento {attempt_no} (subobjetivo {self.current_sub_index + 1}/{len(self.sub_objectives)} - {self.objective_type}): {action.description or action.kind}",
                    )
                    result = self._execute_action(action)
                    self._emit("phase_changed", "Evaluando…")
                    evaluation = self._evaluate_goal(self.current_objective, result)
                    result.status = evaluation.status
                    result.evidence.extend(evaluation.evidence)
                    self.state.evaluation = evaluation
                    self.state.add_decision(f"Intento {attempt_no}: {result.evaluation}")
                    self.state.save()
                    if self.objective_type == "MODIFY" and any("PROPOSAL_ACTION" in ev for ev in result.evidence):
                        continue
                    if evaluation.status == "SUCCESS":
                        sub_success = True
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
                        sub_success = True
                        break
                if not sub_success:
                    break
                if self._has_next_subobjective():
                    self.state.add_decision("Subobjetivo cumplido; avanzando al siguiente")
                    self._advance_subobjective()
                    continue
                self.state.add_decision("No haré nada porque el objetivo ya está cumplido")
                self._emit("success_detected", "Evidencia suficiente de cumplimiento del objetivo")
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
        if self.objective_type == "INSPECT" and attempt_no == 1:
            target = self._select_inspect_target()
            if target:
                return Action(
                    kind="read",
                    description=f"Leer {target.name}",
                    path=str(target.relative_to(self.root)),
                )
        if self.objective_type == "MODIFY" and not self.current_targets:
            target = self._select_inspect_target()
            if target:
                return Action(
                    kind="read",
                    description=f"Leer {target.name} para identificar target",
                    path=str(target.relative_to(self.root)),
                )
            return Action(kind="none", description="No target file specified for MODIFY")
        if self.objective_type == "MODIFY" and not self.modification_proposed:
            proposal = self._propose_modification()
            if proposal:
                return Action(kind="propose", description="Propose modifications", content=proposal)
        if self.objective_type == "MODIFY" and self.modification_proposed and not self.modification_approved:
            if not self._request_modify_approval():
                return Action(kind="none", description="Modification not approved")
        if self.objective_type == "MODIFY" and self.modification_proposed and self.modification_approved:
            write_action = self._apply_approved_modification()
            if write_action:
                return write_action
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
            proposed = Action.from_model(raw_action)
            gated = self._gate_action_by_type(proposed)
            if self.objective_type == "CREATE_ARTIFACT" and gated.kind == "none":
                return self._default_create_action()
            if self.objective_type == "CREATE_ARTIFACT" and gated.kind == "write" and not gated.content:
                return self._fill_write_action(gated)
            return gated
        if self.objective_type == "CREATE_ARTIFACT":
            return self._default_create_action()
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
                message, diff_text = tools.safe_write(str(Path(self.root, action.path)), action.content, self.confirm_cfg)
                self.state.record_file(action.path)
                if diff_text:
                    self._record_modification(action.path, diff_text)
                if self.objective_type == "CREATE_ARTIFACT":
                    if self._is_code_artifact(action.path) or self._matches_create_target(action.path):
                        self.create_code_written = True
                        self.state.create_code_written = True
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
        if action.kind == "read" and action.path:
            target = Path(self.root, action.path)
            try:
                content = tools.read_file(str(target), start=1, lines=400)
                lines = content.strip().splitlines()
                snippet = "\n".join(lines[:5]) if lines else ""
                self._last_inspected_name = target.name
                self._last_inspected_snippet = snippet[:400] if snippet else None
                summary = self._summarize_content(target.name, lines)
                if summary:
                    self.state.semantic_summaries[target.name] = summary
                if self.objective_type == "MODIFY" and not self.current_targets:
                    self.current_targets = [str(target)]
                desc = (
                    f"Read {action.path}; snippet:\n{snippet[:160]}" if snippet else f"Read {action.path}; content inspected"
                )
                evidence = [f"READ_ACTION: read {action.path}", f"SUMMARY: {desc}"]
                note = "Leído sin ejecución"
                return Attempt(
                    action=f"read {action.path}",
                    evaluation=f"{note}. {desc}",
                    status="SUCCESS",
                    stdout=content,
                    evidence=evidence,
                )
            except Exception as exc:  # noqa: BLE001
                return Attempt(
                    action=f"read {action.path}",
                    evaluation=str(exc),
                    status="FAILURE",
                )
        if action.kind == "propose" and action.content:
            self.modification_proposed = True
            target_label = self.current_targets[0] if self.current_targets else "unknown target"
            self.state.modification_summaries[target_label] = action.content
            return Attempt(
                action="propose modification",
                evaluation="Propuesta de modificación generada",
                status="PARTIAL",
                evidence=["PROPOSAL_ACTION"],
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
        read_commands = {"cat", "type", "more", "less", "sed", "awk"}
        if command and (command[0] in read_commands or "cat " in " ".join(command)):
            evidence.append("READ_ACTION: comando de lectura ejecutado")
        if stdout:
            evidence.append(f"stdout observado ({min(len(stdout),80)} chars)")
        if stderr:
            evidence.append(f"stderr observado ({min(len(stderr),80)} chars)")
            evidence.append(f"stderr sample: {stderr[:80].strip()}")
        return evaluation, status, evidence

    def _evaluate_goal(self, objective: str, attempt: Attempt) -> Evaluation:
        evidence = list(attempt.evidence)
        status = attempt.status
        if self.objective_type == "INSPECT":
            read_evidence = any("READ_ACTION" in ev or "stdout observado" in ev for ev in evidence)
            if read_evidence:
                status = "SUCCESS"
                evidence.append("Contenido inspeccionado para el objetivo de lectura")
            else:
                status = "PARTIAL" if status != "FAILURE" else status
                evidence.append("No se ha leído ningún archivo relevante")
        elif self.objective_type == "CREATE_ARTIFACT" and status != "FAILURE":
            artifact_signal = False
            new_files = self.state.files_created[self._sub_files_created_base :]
            if self.create_requires_code:
                if self._has_created_code_file(new_files):
                    artifact_signal = True
                    evidence.append(f"Archivos de código creados: {', '.join(new_files[-3:])}")
                else:
                    status = "PARTIAL" if status != "FAILURE" else status
                    if new_files:
                        evidence.append(f"Archivos creados sin código: {', '.join(new_files[-3:])}")
                    evidence.append("Falta un archivo de código para cumplir el objetivo")
            elif new_files:
                artifact_signal = True
                evidence.append(f"Archivos creados: {', '.join(new_files[-3:])}")
            if attempt.action.startswith("write") or "WRITE_ACTION" in " ".join(evidence):
                if not self.create_requires_code or self.create_code_written:
                    artifact_signal = True
                    evidence.append("Acción de escritura realizada")
            if artifact_signal:
                status = "SUCCESS"
                evidence.append("Objetivo de creación cumplido (artefacto generado)")
            else:
                status = "PARTIAL" if status != "FAILURE" else status
                if attempt.action.startswith("write") or "WRITE_ACTION" in " ".join(evidence):
                    evidence.append("Escritura realizada, pero falta el archivo solicitado")
                else:
                    evidence.append("Sin evidencia de escritura; creación pendiente")
        if attempt.status == "SUCCESS":
            if "test" in attempt.action.lower():
                status = "SUCCESS"
                evidence.append("Tests relacionados con el objetivo pasaron")
            elif self.objective_type not in {"CREATE_ARTIFACT", "INSPECT"}:
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
        if self.objective_type == "INSPECT" and status == "SUCCESS":
            note = "He leído el archivo sin ejecutarlo." if self.no_execute_requested else "He leído el archivo para describirlo."
            rationale = rationale + f" {note}"
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
        file_tokens = [".md", ".py", ".txt", "readme", "documentacion", "documentación", "doc"]
        create_verbs = [
            "crear",
            "crea",
            "crees",
            "genera",
            "generar",
            "escribe",
            "escribir",
            "generate",
            "write",
            "add file",
        ]
        create_nouns = ["script", "archivo", "fichero", "file"]
        modify_keywords = ["modifica", "modificar", "refactor", "ajusta", "update", "cambia"]
        exec_keywords = ["ejecuta", "run", "ejecutar", "execute"]

        create_match = any(k in text for k in create_verbs) or (
            any(n in text for n in create_nouns) and any(v in text for v in create_verbs)
        )
        modify_match = any(k in text for k in modify_keywords)
        inspect_keywords = [
            "que hace",
            "¿que hace",
            "de que trata",
            "explícame",
            "explicame",
            "describe",
            "analiza",
            "inspecciona",
            "read",
            "explain",
            "what does",
            "inspect",
            "lee",
            "list",
            "consulta",
        ]

        if modify_match and any(tok in text for tok in file_tokens):
            return "MODIFY"
        if create_match and any(tok in text for tok in file_tokens):
            return "CREATE_ARTIFACT"
        if create_match:
            return "CREATE_ARTIFACT"
        if any(k in text for k in exec_keywords):
            return "EXECUTE"
        if modify_match:
            return "MODIFY"
        if any(k in text for k in inspect_keywords):
            return "INSPECT"
        return "UNKNOWN"

    def _infer_targets(self, objective: str, objective_type: str) -> List[str]:
        tokens = objective.replace(",", " ").split()
        targets = []
        for tok in tokens:
            if "." in tok:
                candidate = self.root / tok
                if candidate.exists() and candidate.is_file():
                    targets.append(str(candidate))
        if objective_type in {"INSPECT", "MODIFY"} and not targets:
            single = self._single_relevant_file()
            if single:
                targets.append(str(single))
        if objective_type == "CREATE_ARTIFACT" and not targets:
            for tok in tokens:
                if tok.lower().endswith(".md") or tok.lower().endswith(".txt"):
                    targets.append(tok)
                    break
            if not targets:
                targets.append("README.md")
        return targets

    def _mentions_no_execute(self, objective: str) -> bool:
        text = objective.lower()
        terms = ["no lo ejecutes", "sin ejecutar", "solo lee", "no ejecutar", "do not run", "just read"]
        return any(t in text for t in terms)

    def _set_current_subobjective(self, index: int) -> None:
        self.current_sub_index = index
        raw_text = self._sub_objectives_text[index]
        self.current_objective = raw_text
        self.objective_type = self._infer_objective_type(raw_text)
        self.objective_scope_here = self._mentions_here(raw_text)
        self.no_execute_requested = self._mentions_no_execute(raw_text)
        self.current_targets = self._infer_targets(raw_text, self.objective_type)
        self.modification_proposed = False
        self.modification_approved = False
        self.create_requires_code = False
        self.create_primary_target = None
        self.create_code_written = False
        self.create_doc_requested = False
        if self.objective_type == "CREATE_ARTIFACT":
            requires_code, doc_requested, tokens = self._infer_create_context(raw_text)
            self.create_requires_code = requires_code
            self.create_doc_requested = doc_requested
            self.create_primary_target = self._infer_create_target(tokens, requires_code, doc_requested)
            self.state.create_requires_code = self.create_requires_code
            self.state.create_doc_requested = self.create_doc_requested
            self.state.create_primary_target = self.create_primary_target
            self.state.create_code_written = False
        else:
            self.state.create_requires_code = False
            self.state.create_doc_requested = False
            self.state.create_primary_target = None
            self.state.create_code_written = False
        self.state.plan = []
        self.state.evaluation = Evaluation()
        self._sub_files_created_base = len(self.state.files_created)
        self._last_inspected_name = None
        self._last_inspected_snippet = None
        if index == 0:
            self.state.semantic_summaries.clear()
            self.state.project_understanding = ""
            self.state.project_understanding_structured = {}
            self.state.files_modified = []
            self.state.modification_summaries = {}

    def _parse_sub_objectives(self, objective: str) -> List[str]:
        text = objective.replace(";", " y ")
        connectors = [" y ", " and ", " además ", " tambien ", " también ", " y luego ", " y después ", " y despues "]
        parts: List[str] = [objective]
        for conn in connectors:
            if conn in text.lower():
                raw_parts = [p.strip() for p in text.split(conn) if p.strip()]
                if len(raw_parts) > 1:
                    head = raw_parts[0]
                    tail = conn.join(raw_parts[1:]).strip()
                    parts = [head, tail] if tail else [head]
                    break
        return parts or [objective]

    def _has_next_subobjective(self) -> bool:
        return self.current_sub_index + 1 < len(self.sub_objectives)

    def _advance_subobjective(self) -> None:
        if not self._has_next_subobjective():
            return
        self.current_sub_index += 1
        self._set_current_subobjective(self.current_sub_index)
        self._emit("intention_set", f"Nuevo subobjetivo: {self.current_objective}")

    def _mentions_here(self, objective: str) -> bool:
        text = objective.lower()
        here_terms = ["esta carpeta", "aquí", "aqui", "carpeta actual", "esta ruta", "here", "this folder", "current folder"]
        return any(term in text for term in here_terms)

    def _single_relevant_file(self) -> Optional[Path]:
        ignored_dirs = {".git", "__pycache__", ".opencode", ".venv", "venv", "node_modules"}
        candidates: List[Path] = []
        for item in self.root.iterdir():
            if item.is_dir():
                if item.name in ignored_dirs:
                    continue
                continue
            if item.suffix in {".pyc", ".pyo", ".so", ".dylib", ".dll"}:
                continue
            candidates.append(item)
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _select_inspect_target(self) -> Optional[Path]:
        text = self.current_objective.lower()
        tokens = text.replace(",", " ").split()
        for tok in tokens:
            if "." in tok:
                candidate = self.root / tok
                if candidate.exists() and candidate.is_file():
                    return candidate
        if "script" in tokens or "script" in text:
            for item in self.root.iterdir():
                if item.is_file() and "script" in item.name.lower():
                    return item
        return self._single_relevant_file()

    def _default_create_action(self) -> Action:
        text = self.current_objective.lower()
        requires_code, doc_requested, tokens = self._infer_create_context(text)
        target = self.create_primary_target or self._infer_create_target(tokens, requires_code, doc_requested)
        file_scope_tokens = {"script", "archivo", "fichero", "file"}
        file_exts = {".py", ".sh", ".js", ".ts", ".go", ".rs"}
        focus_name = None
        for tok in tokens:
            if any(tok.endswith(ext) for ext in file_exts):
                focus_name = tok
                break
        if not focus_name:
            single = self._single_relevant_file()
            if single:
                focus_name = single.name

        is_file_scope = bool(focus_name or any(w in file_scope_tokens for w in tokens))

        languages = sorted(self.state.languages) if self.state.languages else []
        key_files = self.state.key_files[:5]
        central = [c for c in self.state.central_files if not c.lower().startswith("no se detecta")] if self.state.central_files else []
        summaries = self.state.semantic_summaries
        structured = self.state.project_understanding_structured

        if not target.lower().startswith("readme"):
            content = self._generate_code_artifact(target)
            return Action(
                kind="write",
                description=f"Crear {target}",
                path=target,
                content=content,
            )

        description: List[str] = []
        if is_file_scope and focus_name:
            description.append("## Overview")
            description.append(f"This README documents the file `{focus_name}` based on inspected content.")
        elif is_file_scope:
            description.append("## Overview")
            description.append("This README documents the referenced script based on inspected content.")
        else:
            description.append("## Overview")
            description.append("This README summarizes the repository using observable evidence.")

        if structured:
            behavior = structured.get("behavior", "").strip()
            purpose = structured.get("purpose", "").strip()
            description.append("\n## What it does")
            description.append(behavior or "No clear behavior summary available from inspected evidence.")
            description.append("\n## What it is for")
            description.append(purpose or "No clear purpose could be inferred from inspected evidence.")
        elif summaries:
            description.append("\n## What it does")
            if is_file_scope and focus_name and focus_name in summaries:
                description.append(summaries[focus_name])
            else:
                description.append("Summary based on inspected files:")
                for name, summary in list(summaries.items())[:5]:
                    description.append(f"- {name}: {summary}")
            description.append("\n## What it is for")
            description.append("Purpose is inferred conservatively from inspected files; no execution performed.")
        elif self.state.project_understanding:
            description.append("\n## What it does")
            description.append(self.state.project_understanding)
            description.append("\n## What it is for")
            description.append("Purpose could not be derived beyond the inspected description.")

        if languages or key_files or central:
            description.append("\n## Project context")
            if languages:
                description.append(f"Languages: {', '.join(languages)}.")
            if key_files:
                description.append("Key files:")
                description.extend([f"- {kf}" for kf in key_files])
            if central:
                description.append("Likely central files:")
                description.extend([f"- {c}" for c in central[:5]])
        else:
            description.append("\n## Project context")
            description.append("Limited structural evidence beyond inspected files.")

        description.append("\nNote: This documentation is based on static inspection; no code was executed.")

        title = "# README" if target.lower().startswith("readme") else f"# {target}"
        content_lines = [title, "", *description]
        return Action(
            kind="write",
            description=f"Crear {target}",
            path=target,
            content="\n".join(content_lines) + "\n",
        )

    def _fill_write_action(self, action: Action) -> Action:
        if action.content:
            return action
        default = self._default_create_action()
        return Action(
            kind="write",
            description=action.description or default.description,
            path=action.path or default.path,
            content=default.content,
        )

    def _infer_create_context(self, objective: str) -> tuple[bool, bool, List[str]]:
        text = objective.lower()
        tokens = text.replace(",", " ").split()
        doc_tokens = {"readme", "documenta", "documentar", "documentacion", "documentación", "doc", "docs"}
        code_tokens = {"script", "programa", "herramienta", "tool", "cli", "app", "aplicacion", "aplicación", "servicio"}
        language_tokens = {
            "python",
            "py",
            "javascript",
            "js",
            "node",
            "typescript",
            "ts",
            "bash",
            "shell",
            "sh",
            "go",
            "rust",
        }
        has_doc_request = any(tok in doc_tokens for tok in tokens) or "readme" in text
        has_code_request = any(tok in code_tokens for tok in tokens) or any(tok in language_tokens for tok in tokens)
        has_code_ext = any(tok.endswith(ext) for tok in tokens for ext in (".py", ".js", ".ts", ".sh", ".go", ".rs"))
        requires_code = has_code_request or has_code_ext
        doc_requested = has_doc_request and not requires_code
        return requires_code, doc_requested, tokens

    def _infer_create_target(self, tokens: List[str], requires_code: bool, doc_requested: bool) -> str:
        for tok in tokens:
            if tok.endswith(".py") or tok.endswith(".js") or tok.endswith(".ts") or tok.endswith(".sh"):
                return tok
            if tok.endswith(".md") or tok.endswith(".txt"):
                if not requires_code:
                    return tok
        if doc_requested or "readme" in tokens or "documentar" in tokens or "documentacion" in tokens or "documentación" in tokens:
            return "README.md"
        if "python" in tokens:
            return "script.py"
        if "javascript" in tokens or "node" in tokens:
            return "script.js"
        if "bash" in tokens or "shell" in tokens:
            return "script.sh"
        if requires_code:
            return "script.py"
        return "README.md"

    def _generate_code_artifact(self, target: str) -> str:
        payload = {
            "objective": self.current_objective,
            "target": target,
            "constraints": [
                "Generate a minimal, correct implementation",
                "No execution, no external dependencies",
                "Keep it short and readable",
            ],
        }
        system = (
            "Generate the full file content for the requested code artifact. "
            "Return only code, no Markdown. Keep it minimal and aligned with the objective."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = backend.call_llama(
                self.backend.settings.base_url,
                {"model": self.model, "messages": messages, "temperature": 0.2},
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() + "\n" if content else f"# TODO: implement {target}\n"
        except Exception:
            return f"# TODO: implement {target}\n"

    def _ensure_semantic_synthesis(self) -> None:
        if self.state.project_understanding:
            return
        scope = "file" if len(self.state.semantic_summaries) == 1 else "repository"
        payload = {
            "scope": scope,
            "user_intent": self.current_objective,
            "explanation_intent": self._explanation_intent(self.current_objective),
            "files_inspected": self.state.semantic_summaries,
            "languages": sorted(self.state.languages),
            "structure": self.state.directories[:20],
            "key_files": self.state.key_files[:10],
            "central_files": self.state.central_files[:10],
            "constraints": ["No execution performed", "Only inspected content"],
        }
        system = (
            "Eres un sintetizador semántico para documentación humana. Devuelve JSON con claves 'behavior' y 'purpose'. "
            "Behavior: solo acciones observables directamente en el contenido inspeccionado (sin suposiciones). "
            "Purpose: hipótesis prudente usando lenguaje suave ('puede servir para', 'parece pensado para') y sin introducir acciones nuevas. "
            "No uses jerga interna, no incluyas snippets ni detalles forenses. Basado estrictamente en la evidencia." 
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = backend.call_llama(self.backend.settings.base_url, {"model": self.model, "messages": messages, "temperature": 0.2})
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                parsed = None
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and (parsed.get("behavior") or parsed.get("purpose")):
                    self.state.project_understanding_structured = {
                        "behavior": str(parsed.get("behavior", "")).strip(),
                        "purpose": str(parsed.get("purpose", "")).strip(),
                    }
                    combined = []
                    if parsed.get("behavior"):
                        combined.append(f"Behavior: {parsed.get('behavior')}")
                    if parsed.get("purpose"):
                        combined.append(f"Purpose: {parsed.get('purpose')}")
                    self.state.project_understanding = " ".join(combined).strip()
                else:
                    self.state.project_understanding = content.strip()
        except Exception:
            pass

    def _summarize_content(self, name: str, lines: List[str]) -> str:
        if not lines:
            return "No observable content (empty read)."
        text = "\n".join(lines[:200]).lower()
        statements: List[str] = []
        statements.append(
            f"Este archivo (`{name}`) fue leído de forma estática; la explicación se basa solo en el texto, sin ejecución."
        )

        def has_any(tokens):
            return any(tok in text for tok in tokens)

        if has_any(["print"]):
            statements.append("Imprime información en la consola.")
        if has_any(["datetime", "time"]):
            statements.append("Usa utilidades de fecha y hora.")
        if has_any(["input("]):
            statements.append("Solicita entrada del usuario por consola.")
        if has_any(["argparse", "sys.argv"]):
            statements.append("Lee argumentos de línea de comandos.")
        if has_any(["click", "typer"]):
            statements.append("Expone una interfaz de línea de comandos basada en comandos.")
        if has_any(["subparsers", "add_argument"]):
            statements.append("Define opciones o comandos CLI configurables.")
        if has_any(["requests", "http", "urllib", "fetch"]):
            statements.append("Incluye llamadas de red/HTTP explícitas.")
        if has_any(["socket", "websocket"]):
            statements.append("Trabaja con conexiones de red persistentes.")
        if has_any(["open(", "with open"]):
            statements.append("Abre archivos desde el sistema de archivos.")
        if has_any(["read(", "write("]):
            statements.append("Realiza operaciones de lectura/escritura en archivos.")
        if has_any(["json", "yaml", "toml"]):
            statements.append("Manipula datos en formatos estructurados (JSON/YAML/TOML).")
        if has_any(["csv", "pandas"]):
            statements.append("Procesa datos tabulares o CSV.")
        if has_any(["sqlite", "postgres", "mysql", "sqlalchemy"]):
            statements.append("Interacciona con bases de datos.")
        if has_any(["class ", "def "]):
            statements.append("Define funciones o clases reutilizables.")
        if has_any(["async ", "await", "asyncio"]):
            statements.append("Incluye lógica asíncrona.")
        if has_any(["logging", "logger"]):
            statements.append("Registra eventos o mensajes de ejecución.")
        if has_any(["add", "subtract", "multiply", "divide", "calculator", "calculadora"]):
            statements.append("Incluye operaciones aritméticas o lógica de calculadora.")
        if has_any(["if __name__ == '__main__'", "main("]):
            statements.append("Incluye un bloque de entrada para ejecución directa.")

        if len(statements) == 1:
            statements.append("No se observan acciones explícitas más allá de la estructura básica.")

        return " ".join(statements)

    def _record_modification(self, path: str, diff_text: str) -> None:
        if not diff_text.strip():
            return
        normalized = path
        if normalized not in self.state.files_modified:
            self.state.files_modified.append(normalized)
        summary = self._summarize_diff(path, diff_text)
        if summary:
            self.state.modification_summaries[normalized] = summary

    def _summarize_diff(self, path: str, diff_text: str) -> str:
        added = []
        removed = []
        for line in diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added.append(line[1:].strip())
            elif line.startswith("-"):
                removed.append(line[1:].strip())
        if not added and not removed:
            return ""
        focus = f"Updated {Path(path).name}. "
        detail = []
        if added:
            detail.append(f"Added {len(added)} line(s)")
        if removed:
            detail.append(f"Removed {len(removed)} line(s)")
        return focus + ", ".join(detail) + "."

    def _request_modify_approval(self) -> bool:
        if self.modification_approved:
            return True
        target = Path(self.current_targets[0]).name if self.current_targets else "unknown"
        proposal = self.state.modification_summaries.get(self.current_targets[0], "")
        if proposal:
            print("Proposed changes:")
            print(proposal)
        approved = tools.confirm(f"Apply proposed modifications to {target}?", self.confirm_cfg)
        self.modification_approved = approved
        return approved

    def _propose_modification(self) -> Optional[str]:
        if not self.current_targets:
            return None
        target = Path(self.current_targets[0]).name
        summary = self.state.semantic_summaries.get(target)
        if not summary:
            return None
        payload = {
            "target": target,
            "objective": self.current_objective,
            "semantic_summary": summary,
            "constraints": ["No execution", "Propose changes only"],
        }
        system = (
            "Generate a concise list of proposed modifications based on the objective and summary. "
            "Do not write code. Output 2-5 bullet points in plain text, each describing what to change and why."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = backend.call_llama(
                self.backend.settings.base_url,
                {"model": self.model, "messages": messages, "temperature": 0.2},
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or None
        except Exception:
            return None

    def _apply_approved_modification(self) -> Optional[Action]:
        if not self.current_targets:
            return None
        target = Path(self.current_targets[0])
        try:
            original = target.read_text(encoding="utf-8")
        except Exception:
            return None
        proposal = self.state.modification_summaries.get(self.current_targets[0])
        if not proposal:
            return None
        for attempt in range(2):
            patch = self._generate_patch(target.name, original, proposal)
            if not patch:
                continue
            updated = self._apply_unified_diff(original, patch)
            if not updated:
                continue
            if self._validate_patch(original, updated):
                return Action(
                    kind="write",
                    description=f"Apply approved modifications to {target.name}",
                    path=str(target),
                    content=updated,
                )
        return None

    def _generate_patch(self, filename: str, original: str, proposal: str) -> Optional[str]:
        payload = {
            "objective": self.current_objective,
            "target": filename,
            "proposal": proposal,
            "original": original,
            "constraints": [
                "Apply only the approved changes",
                "Do not add new features",
                "Preserve all unrelated content",
                "Return unified diff only",
            ],
        }
        system = (
            "Generate a unified diff patch for the target file. "
            "Only include necessary hunks. Do not output explanations or extra text."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = backend.call_llama(
                self.backend.settings.base_url,
                {"model": self.model, "messages": messages, "temperature": 0.2},
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() or None
        except Exception:
            return None

    def _apply_unified_diff(self, original: str, diff_text: str) -> Optional[str]:
        if "@@" not in diff_text:
            return None
        orig_lines = original.splitlines(keepends=True)
        new_lines: List[str] = []
        orig_idx = 0
        lines = diff_text.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("@@"):
                header = line
                try:
                    old_part = header.split(" ")[1]
                    old_start = int(old_part.split(",")[0][1:]) - 1
                except Exception:
                    return None
                if old_start < orig_idx:
                    return None
                new_lines.extend(orig_lines[orig_idx:old_start])
                orig_idx = old_start
                idx += 1
                while idx < len(lines) and not lines[idx].startswith("@@"):
                    hunk_line = lines[idx]
                    if hunk_line.startswith("+"):
                        new_lines.append(hunk_line[1:] + "\n")
                    elif hunk_line.startswith("-"):
                        if orig_idx >= len(orig_lines):
                            return None
                        if orig_lines[orig_idx].rstrip("\n") != hunk_line[1:]:
                            return None
                        orig_idx += 1
                    elif hunk_line.startswith(" "):
                        if orig_idx >= len(orig_lines):
                            return None
                        if orig_lines[orig_idx].rstrip("\n") != hunk_line[1:]:
                            return None
                        new_lines.append(orig_lines[orig_idx])
                        orig_idx += 1
                    idx += 1
                continue
            idx += 1
        new_lines.extend(orig_lines[orig_idx:])
        return "".join(new_lines)

    def _validate_patch(self, original: str, updated: str) -> bool:
        if not updated.strip():
            return False
        if original == updated:
            return False
        import difflib

        ratio = difflib.SequenceMatcher(None, original, updated).ratio()
        if ratio < 0.85:
            return False
        if "\n\n\n\n" in updated:
            return False
        if updated.count("#!/") > 1:
            return False
        return self._validate_format(updated)

    def _validate_format(self, updated: str) -> bool:
        try:
            import ast

            ast.parse(updated)
            return True
        except Exception:
            pass
        try:
            import json as jsonlib

            jsonlib.loads(updated)
            return True
        except Exception:
            pass
        return True

    def _explanation_intent(self, objective: str) -> str:
        text = objective.lower()
        if any(k in text for k in ["readme", "documenta", "documentar", "documentacion", "documentación"]):
            return "Explain the purpose and behavior for a reader new to the project."
        if any(k in text for k in ["explica", "explícame", "describe", "que hace", "de que trata"]):
            return "Explain what the code does in simple terms."
        return "Summarize the inspected evidence in human language."

    def _gate_action_by_type(self, action: Action) -> Action:
        # INSPECT: allow only read; convert benign run->read when possible; avoid execution/ls
        if self.objective_type == "INSPECT":
            if action.kind == "run":
                target = action.path or None
                if not target:
                    single = self._single_relevant_file()
                    if single:
                        target = str(single.relative_to(self.root))
                return Action(kind="read", description=action.description or "Leer para inspeccionar", path=target)
            if action.kind not in {"read"}:
                single = self._single_relevant_file()
                return Action(kind="read", description="Leer archivo para inspeccionar", path=str(single.relative_to(self.root)) if single else action.path)
            return action
        # CREATE_ARTIFACT: only write/diff-oriented actions are meaningful; block inspection commands
        if self.objective_type == "CREATE_ARTIFACT":
            if action.kind in {"read", "run"}:
                return Action(kind="none", description="Esperando plan de escritura")
            if action.kind == "write" and self.create_requires_code and not self.create_code_written:
                if not action.path:
                    return Action(kind="none", description="Esperando creación del archivo de código solicitado")
                if not self._matches_create_target(action.path) and not self._is_code_artifact(action.path):
                    return Action(kind="none", description="Esperando creación del archivo de código solicitado")
            return action
        if self.objective_type == "MODIFY":
            if action.kind != "write":
                return Action(kind="none", description="Esperando modificación con escritura")
            if action.path and self.current_targets:
                target_names = {Path(t).name for t in self.current_targets}
                if Path(action.path).name not in target_names:
                    return Action(kind="none", description="Write target does not match MODIFY target")
        return action

    def _matches_create_target(self, path: str) -> bool:
        if not self.create_primary_target:
            return False
        return Path(path).name == Path(self.create_primary_target).name

    def _is_code_artifact(self, path: str) -> bool:
        return Path(path).suffix.lower() in {".py", ".js", ".ts", ".tsx", ".sh", ".go", ".rs", ".rb", ".php", ".java", ".cs"}

    def _has_created_code_file(self, created_files: List[str]) -> bool:
        if not created_files:
            return False
        if self.create_primary_target:
            target_name = Path(self.create_primary_target).name
            if any(Path(path).name == target_name for path in created_files):
                return True
        return any(self._is_code_artifact(path) for path in created_files)


class _NoOpEvents:
    def emit(self, event: str, message: str) -> None:  # noqa: D401
        """No-op event sink."""
        return None
