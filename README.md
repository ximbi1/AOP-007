# AI-assisted Development CLI (Prototype)

This repository contains a terminal-first assistant designed to operate inside any local workspace. It mirrors the spirit of tools like OpenCode: understand what is in the folder, plan work with the user, and only apply changes after explicit confirmation.

## Features

- Interactive CLI with a small REPL (`omoc`) to inspect and modify projects cautiously.
- Iterative agent mode: objetivo → plan → acción → evaluación → corrección with explicit attempts and state tracking.
- Ligera memoria persistente: `ProjectState` guarda lenguajes, frameworks, archivos clave y decisiones en `.opencode/state.json` (carga automática si existe).
- Built-in tools: list files, read file segments, regex search, safe writes with diffs, and guarded command execution.
- Workspace analysis to detect languages, frameworks, and missing project scaffolding.
- Plan-before-action workflow: propose steps, request confirmation, and execute sequentially.
- Git helpers: initialize repositories, show status/diffs, create automatic commits, and revert.

## Quickstart

```bash
# Help
python -m opencode_agent.cli --help

# Interactive mode
python -m opencode_agent.cli interactive

# Analyze current workspace
python -m opencode_agent.cli analyze .

# Iterative agent loop (starts local llama-server if needed)
python -m opencode_agent.cli agent "tu objetivo" --max-attempts 3

# (Optional) use the bundled launcher, add it to PATH and run
./opencode agent "tu objetivo"
```

Interactive mode exposes commands like `list`, `read`, `search`, `write`, `plan`, `run`, `agent`, and `git` helpers. Every write or command execution requests confirmation unless `--yes` is provided.

## Design Principles

- Never modify files without explicit user approval.
- Always show a plan and a diff before applying changes.
- Be transparent about detected context (languages, frameworks, missing pieces).
- Keep session state in memory to remember recent decisions.
- Separate phases: analysis → planning → execution → evaluation → correction; capture stdout/stderr from commands and retry with hypotheses when something fails.
- Decide cuándo parar: si el objetivo ya está cumplido, si se agotaron intentos o si se requiere decisión humana.
- Comprensión semántica ligera (heurística): identifica posibles entrypoints, archivos centrales y configs con niveles de confianza (alta/media/baja). Siempre se presenta como hipótesis y requiere confirmación humana.

## Notes

This is an initial prototype focused on safety and clarity. Extend the command set or plug different model backends by expanding the planner and executor stubs in `opencode_agent/`.

## Local model backend

- The agent auto-starts a `llama-server` (llama.cpp) using the defaults baked into `opencode_agent/backend.py`:
  - Binary: `/home/ximbi/MODELOSIA/llama.cpp/build/bin/llama-server`
  - Model: `~/MODELOSIA/models/granite/granite-4.0-h-1b-Q4_0.gguf`
  - Arguments: `-t 10 -c 25096 -b 512 -ngl 0 --temp 0.2 --top_k 80 --top_p 0.9 --repeat_penalty 1.05 --mirostat 0 --port 8080 --host 0.0.0.0`
- The launcher sets `LD_LIBRARY_PATH` to `/opt/intel/oneapi/mkl/2025.3/lib/intel64` for the server process.
- Logs live at `~/.cache/opencode/llama_server.log`.

## ProjectState (memoria ligera)

- Mantiene snapshot del workspace: lenguajes, frameworks, archivos clave, estructura superficial y decisiones/hypótesis recientes.
- Se carga automáticamente desde `.opencode/state.json` si existe y se actualiza tras cada intento con evidencia útil.
- No realiza análisis profundo; solo conserva contexto para próximas ejecuciones.

## Comprensión semántica ligera

- Componente `ProjectUnderstanding` genera pistas opcionales usando heurísticas simples: nombres comunes (`main.*`, `app.*`, `index.*`, `server.*`, `cli.*`), configs conocidas (`package.json`, `pyproject.toml`, etc.), carpetas típicas (`src/`, `app/`, `tests/`) y referencias por imports (regex, sin AST).
- Cada pista incluye nivel de confianza (alta/media/baja) y una razón explicable; nunca se asume como verdad absoluta.
- El agente usa estas pistas solo para justificar por qué prioriza un archivo o solicita confirmación; puede decir “no tengo suficiente certeza” y pedir guía humana.

## Evaluación por objetivos

- Tras cada intento, el agente responde explícitamente “¿qué evidencia tengo de que el objetivo se ha cumplido?”
- Estados posibles: `SUCCESS` (evidencia suficiente), `PARTIAL` (progreso, pero incompleto), `FAILURE` (sin evidencia suficiente).
- Basado en observables: stdout/stderr, códigos de salida, archivos creados/modificados.
- Con evidencia insuficiente, puede elegir estrategias: cambiar enfoque, simplificar, pedir decisión humana o detenerse.

## Cuándo el agente decide parar

- Si detecta que el objetivo ya está cumplido: declara éxito y no ejecuta más acciones.
- Si alcanza el límite de intentos sin evidencia suficiente: recomienda parar o solicitar intervención humana.
- Puede optar por no hacer nada cuando la evidencia indica que no se requieren más pasos.

## Tests y contratos de comportamiento

- Suite de `pytest` que valida conductas clave: no actuar si el objetivo ya está cumplido, recomendar parar tras varios fallos, pedir decisión humana ante ambigüedad y exigir confirmaciones para acciones riesgosas.
- Tests de `ProjectState` garantizan persistencia determinista de decisiones, planes y evaluaciones en `.opencode/state.json`.
- Tests de seguridad verifican que no se ejecuten comandos ni escrituras sin confirmación ni se sobrescriban archivos sin diffs previos.
- Ejecuta `pytest` en el repositorio para validar estos contratos antes de usar el agente.
