# AI-assisted Development CLI (Prototype)

This repository delivers a terminal-first development assistant inspired by OpenCode. It understands your workspace, plans with you, executes cautiously, streams concise status, and keeps a traceable memory—always prioritizing safety, humility, and human control. The interactive CLI is branded as **Cortex Agent Shell**.

## Why it stands out

- **Structured agent loop**: objective → plan → action → evaluation → correction, with explicit attempts and bounded retries.
- **Real-time observability**: console events for phase changes, intent summaries, proposed/cancelled commands, strategy shifts, stop/success notices—never exposing chain-of-thought.
- **Persistent output log**: every console line is duplicated into `outputlog.txt` with start/end markers for auditing and debugging.
- **Persistent lightweight memory**: `ProjectState` snapshots languages, structural hints, key files, decisions, hypotheses, created files, and evaluations in `.opencode/state.json` for continuity.
- **Heuristic semantic understanding**: strong-evidence framework detection (entry/config/bootstrap + deps), low-confidence hints for weak signals, import-centrality to surface likely core files, and strict filtering of noise (.git, __pycache__, venvs, node_modules, build artifacts, compiled extensions).
- **Goal-aware evaluation**: creation goals are marked SUCCESS when artifacts are written; no forced execution. Observables only (stdout/stderr, exit codes, file changes, tests).
- **Safety contracts**: confirmations before writes/commands, diff previews, security and behavioral tests to prevent reckless actions.
- **Local-first LLM backend**: auto-starts llama.cpp `llama-server` (configurable) so you can run fully offline.
- **Git-aware ergonomics**: status/diff helpers, commit/revert with confirmation.
- **Composite goals handled safely**: multi-part instructions are split into independent sub-goals (inspect/read vs create/modify), each inferred, planned, executed, and evaluated in isolation; global success only when all sub-goals succeed.
- **Semantic understanding as a first-class artifact**: per-file semantic summaries from INSPECT plus a repository-level behavior/purpose synthesis (one LLM call) feed documentation generation; no snippets or code execution required.

## Quickstart

```bash
# Help
python -m opencode_agent.cli --help

# Interactive REPL (Cortex Agent Shell)
python -m opencode_agent.cli interactive

# Analyze current workspace (read-only)
python -m opencode_agent.cli analyze .

# Iterative agent loop (auto-starts local llama-server if needed)
python -m opencode_agent.cli agent "your objective" --max-attempts 3

# Optional launcher: add to PATH and run
./opencode agent "your objective"
```

Interactive mode exposes `list`, `read`, `search`, `write`, `plan`, `run`, `agent`, and git helpers. Every write or command execution requests confirmation unless `--yes` is provided.

## Design principles

- No file modifications without explicit approval; always show plan and diff first.
- Transparent context: languages, structural frameworks, key files, missing scaffolding, low-confidence hints when evidence is weak.
- Clear phases: analysis → planning → execution → evaluation → correction; capture stdout/stderr and retry with hypotheses when needed.
- Stop when appropriate: success detected, attempts exhausted, or human decision required.
- Treat model output as hypotheses; prefer “not clear” over false certainty.
- Respect sub-goal boundaries: each subobjective uses its exact text for inference and evidence; no cross-contamination of types or artifacts.
- Keep interpretation separate: INSPECT gathers evidence, SEMANTIC_SYNTHESIS interprets behavior/purpose, CREATE_ARTIFACT writes docs from that understanding.

## Local model backend

- Defaults (configurable in `opencode_agent/backend.py`):
  - Binary: `/home/ximbi/MODELOSIA/llama.cpp/build/bin/llama-server`
  - Model: `~/MODELOSIA/models/granite/granite-4.0-h-1b-Q4_0.gguf`
  - Args: `-t 10 -c 25096 -b 512 -ngl 0 --temp 0.2 --top_k 80 --top_p 0.9 --repeat_penalty 1.05 --mirostat 0 --port 8080 --host 0.0.0.0`
- Launcher sets `LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2025.3/lib/intel64`.
- Logs: `~/.cache/opencode/llama_server.log`.

## Workspace analysis & semantic hints

- Strict path filtering; no noise from VCS, caches, virtualenvs, node_modules, build/dist, or compiled artifacts.
- Framework detection only with structural evidence; weak signals become low-confidence hints, not facts.
- Centrality heuristic highlights most-imported files; absence of evidence is stated explicitly.
- Reports key files, languages, missing scaffolding, and central files succinctly.
- Composite safety: INSPECT sub-goals stay read-only; CREATE/MODIFY sub-goals focus on diffs/writes with confirmation.

## ProjectState (lightweight memory)

- Snapshots context and decisions in `.opencode/state.json`, auto-loaded and updated per attempt.
- Keeps shallow structure and recent choices; avoids deep AST/LSP analysis by design.

## Goal-oriented evaluation

- Explicit evidence check after each attempt: `SUCCESS`, `PARTIAL`, or `FAILURE` based only on observables.
- Creation goals: SUCCESS when artifacts are generated; no forced execution.
- Strategies when evidence is thin: change approach, simplify, ask human, or stop.
- Composite goals: each sub-goal is evaluated independently; global SUCCESS only when all provide matching evidence (reads for INSPECT, writes for CREATE/MODIFY).

## Agent pipeline (high level)

- **Analysis**: workspace scan with strict filtering; detect languages, key/central files, and structural scaffolding.
- **INSPECT (read-only)**: safe file reads produce per-file semantic summaries (no execution, no snippets needed for downstream); evidence is stored in `ProjectState`.
- **Semantic synthesis**: a single LLM call turns inspected evidence into a behavior/purpose explanation (repo-level), stored structurally (behavior/purpose fields). No code is re-read or executed here.
- **Planning**: per sub-goal (supports composite objectives). Plans are regenerated in isolation for each subobjective using its exact text.
- **Execution**: gated tools per type (INSPECT → read; CREATE/MODIFY → write/diff with confirmation; EXECUTE → run with confirmation). Composite goals advance only when a sub-goal succeeds.
- **Evaluation**: evidence-only SUCCESS/PARTIAL/FAILURE; creation requires real writes, inspection requires reads. Composite SUCCESS only when all sub-goals succeed.
- **Documentation (CREATE_ARTIFACT)**: README/docs are drafted from semantic understanding (file summaries or repo-level synthesis) plus structural signals (languages, key/central files). No code snippets or speculative behavior; explicit note that no execution was performed.
- **Events/UX**: real-time events for phase, intent, strategy, stop/success; final summary remains the authoritative report.

## README generation for CREATE_ARTIFACT

- Default README writes are auto-proposed when a create sub-goal lacks a plan; the target filename is inferred from the request (e.g., README.md).
- Content is evidence-based: uses detected languages, key files, central files, semantic summaries (file-level) or synthesized repo understanding (behavior/purpose); scopes adapt to “single script” vs “repository” wording.
- Language is honest and non-speculative (“based on inspected files”, “no execution performed”); no placeholders or invented behavior.

## Real-time events (UX)

- Concise console events: phases, intents, proposed/cancelled commands, strategy changes, stop/success recommendations.
- Final summary remains as the authoritative report; events are informative, not reasoning dumps.
- Sub-goals are announced with type and index (e.g., subobjetivo 2/2 - CREATE_ARTIFACT) to keep the flow traceable.
- On successful completion, the CLI generates a human closing summary via a single LLM call (no internal terms, just what changed and why it matters).

## Output log

- All CLI output is mirrored to `outputlog.txt` in the current working directory.
- The log includes `=== Cortex Agent Session Start ===` and `=== Cortex Agent Session End ===` markers.
- Useful for traceability, debugging, and reviewing agent actions after the fact.

## Safety & behavioral tests

- `pytest` suite enforces ethics: no action if goal already met, stop recommendation after repeated failures, human prompt on ambiguity, confirmation for risky actions.
- ProjectState tests ensure deterministic persistence; security tests guard against unconfirmed writes/commands and blind overwrites.
- Run `pytest --maxfail=1 --disable-warnings -q` to validate contracts.

## Extending

- Add tools, adjust heuristics, or plug a different model backend by extending `opencode_agent/` components.
- Event sink is pluggable for alternative UIs; default prints to console.
- Keep the core philosophy: cautious execution, evidence-first decisions, and human-in-the-loop control.
