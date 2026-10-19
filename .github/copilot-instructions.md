# Repository instructions for Copilot

## Project Guidelines

- Read `README_CONTEXT.md` and the relevant subsystem documentation before changing behavior.
- Python project; dependencies in `requirements.txt`. Tests are `unittest` in `tests/`.
- Run `python roadmap_agent.py validate` and `python -m unittest discover -s tests -v` before finishing.

## Scope and safety

- Keep changes within the assigned issue and its acceptance criteria; do not implement unrelated roadmap items.
- Make small, focused diffs. Prefer editing existing modules over adding new ones.
- Do not commit credentials, tokens, private datasets, generated databases, or local environment files.
- Never merge a pull request. Open it for human review and say `Fixes <issue URL>`.

## Quality

- Add or update focused tests for every behavior change; tests must not need network or credentials.
- Report tests that could not run, and why, in the pull request description.

## Roadmap swarm

- `ROADMAP.md` items with `Status: ready` are dispatched to cloud agents by `.github/workflows/roadmap-agent.yml` (every 3 hours, capped concurrency, retries after quota resets).
- Custom agents live in `.github/agents/` (`tester`, `reviewer`).
