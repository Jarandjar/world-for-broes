# Repository instructions for Copilot

- Read `README_CONTEXT.md` and the relevant subsystem documentation before changing behavior.
- Keep changes within the assigned issue and its acceptance criteria; do not implement unrelated roadmap items.
- Add or update focused tests for behavior changes. Run `python -m unittest discover -s tests -v` when Python tests are relevant.
- Do not commit credentials, private datasets, generated databases, or local environment files.
- Report tests that could not run and their blockers in the pull request. Never merge a pull request.
