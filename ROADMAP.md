# Coding roadmap

Ideas stay **Proposed** until a maintainer changes their status to **Ready**. When a Ready change reaches `main`, the roadmap workflow creates one GitHub issue and starts a Copilot cloud task. The cloud task is scoped to that issue and creates a pull request for human review; it never merges.

To add an idea, copy the item format below. Give it a stable, unique ID, explain the value, describe the expected outcome, and provide testable acceptance criteria. Keep credentials and sensitive data out of this file. Do not mark an item Ready until its scope and criteria are clear.

## Proposed ideas

### `cloud-task-dashboard` Show Copilot cloud tasks alongside swarm jobs
- Status: ready
- Why: `swarmctl jobs list` shows the local swarm queue, while Copilot cloud task runs have a separate lifecycle and should be easy to check without confusing the two systems.
- Outcome: Operators can see cloud task state and linked branches or pull requests from the existing CLI, while GitHub remains the source of truth.
- Acceptance criteria:
  - List cloud tasks with state, task ID, update time, and any linked pull request or branch.
  - Make the distinction between local swarm jobs and GitHub cloud tasks explicit.
  - Handle missing permissions and API failures with clear errors.

### `persistent-swarm-job-history` Persist swarm job history
- Status: proposed
- Why: In-memory job lists lose their history when the process restarts, limiting troubleshooting and trend analysis.
- Outcome: Operators can inspect recent completed and failed swarm jobs across process restarts.
- Acceptance criteria:
  - Persist job lifecycle transitions without requiring optional vector-search services.
  - Preserve the existing `jobs list` and `jobs inspect` behavior.
  - Cover persistence, restart recovery, and failed-job reporting with tests.

### `retrieval-provenance` Expose context retrieval provenance
- Status: proposed
- Why: Context retrieval is more useful and auditable when users can see which summaries or documents were injected into a Codex request.
- Outcome: Codex responses and logs can identify the retrieved context sources and their relevance.
- Acceptance criteria:
  - Return source identifiers with retrieved context without changing the existing plain-text response by default.
  - Record which sources were used for each request.
  - Test empty, fallback, and vector-search retrieval paths.

### `token-budget-enforcement` Enforce context budgets with token counts
- Status: proposed
- Why: Character-count estimates can misjudge token usage and risk oversized prompts.
- Outcome: Scoped context packs respect a configurable token budget using a tokenizer when available and an explicit fallback otherwise.
- Acceptance criteria:
  - Enforce the requested budget before sending a prompt.
  - Report when optional tokenization is unavailable and identify the fallback estimate.
  - Test trimming order and boundary conditions.

## Item template

Copy these fields under a heading like `### \`my-idea-id\` Short, actionable title`:
- Status: proposed
- Why: Explain the problem and who benefits.
- Outcome: Describe observable completion.
- Acceptance criteria:
  - State a verifiable behavior or test.
