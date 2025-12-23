# Cortex Agent Session Context

## Project overview
- Terminal-based AI development assistant branded **Cortex Agent Shell**.
- Core loop: analysis → planning → execution → evaluation → correction.
- Composite goals split into sub-objectives with strict isolation per sub-goal.
- Safety: confirmations for writes/commands, diffs shown, no execution without consent.

## Key architecture components
- `opencode_agent/agent.py`: IterativeAgent, sub-objective orchestration, MODIFY proposal/approval/write flow.
- `opencode_agent/analyzer.py`: workspace analysis with strict filtering + structural framework detection.
- `opencode_agent/state.py`: persistent `ProjectState` in `.opencode/state.json`.
- `opencode_agent/cli.py`: Cortex Agent Shell UI, colored events, output logging to `outputlog.txt`.

## Semantic understanding layer
- INSPECT reads files and generates per-file semantic summaries stored in `state.semantic_summaries`.
- Semantic synthesis phase (single LLM call) produces repo-level behavior/purpose in `state.project_understanding_structured`.
- README generation consumes semantic understanding (not snippets), with “What it does” and “What it is for” sections.

## MODIFY flow (current)
- INSPECT (if needed to find target) → PROPOSE_MODIFICATION → WAIT_FOR_APPROVAL → APPLY_APPROVED_MODIFICATION → WRITE → EVAL.
- Proposal stored in `state.modification_summaries`.
- Approved modifications applied via patch (unified diff), not full rewrite.
- Validation ensures non-empty output and high overlap with original content.

## Success summary (CLI)
- On SUCCESS, LLM generates a closing report with structure:
  Done / Key actions / Result / Constraints.
- Uses only evidence: inspected files, created files, modified files, structured understanding.
- Dedupes file paths case-insensitively.
- Falls back to deterministic summary if LLM output is invalid.

## UX improvements
- CLI banner: “Cortex Agent Shell”.
- Colored phase/action/status events (ANSI, NO_COLOR supported).
- Diffs colorized (added green, removed red).
- All console output mirrored to `outputlog.txt` with start/end markers.

## Known recent fixes
- Fixed `_Tee.isatty()` to restore phase/event output.
- Handle LLM command parsing errors (unclosed quotes) safely.
- README generation improved to be human, evidence-based, and non-speculative.

## Tests
- `pytest --maxfail=1 --disable-warnings -q` should pass.
