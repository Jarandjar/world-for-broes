# Context Compression & Retrieval Plan

## Goals
1. Prevent Copilot / GPT prompt overflows (>128k tokens).
2. Preserve deep context via structured summaries + retrieval.
3. Provide operators with tooling to scope queries and view summaries.

## Pillars

1. **Context Summaries**
   - Agent: `agent_context_distiller.py`.
   - Walk defined subsystems (governor, Demiurge, evolution, bosses, quests, lore, CLI, docs).
   - Capture ~1k-token markdown summary per subsystem with sections: Purpose, Key Files, Data Sources, Current Status, Open Risks.
   - Persist to DuckDB table `context_summaries` (`subsystem`, `summary_md`, `updated_at`) and JSON in `out/context_summaries/`.
   - Embed summaries using `gpu_embeddings.py` and store vectors in Qdrant collection `context_summaries` for retrieval.
   - Refresh cadence: manual via CLI + nightly automation.

2. **`swarmctl gpt-scope` Command**
   - CLI entrypoint: `swarmctl gpt-scope [--path <glob>] [--topic <name>] [--subsystem <id>] [--include-summaries] [--output <file>]`.
   - Resolves file lists using glob/topic manifest (YAML map maintained in repo, e.g., `configs/gpt_scope_map.yaml`).
   - Pulls latest summary text when `--include-summaries` set.
   - Outputs concatenated content to stdout or file; warns if estimated tokens > 100k (use `tiktoken` or simple char/4 heuristic).
   - Downstream use: pipe into Copilot CLI / GPT request.

3. **Codex Retrieval Wrapper**
   - Extend `packages/shared/src/codexClient.ts` and `python/codex_client.py` with optional `context_query` field.
   - When provided, lookup relevant summaries/documents:
     1. Embed query via GPU helper.
     2. Search Qdrant `context_summaries` + `docs` collections.
     3. Fetch top-N (configurable) summaries.
     4. Chunk them to ≤4k tokens per snippet.
     5. Prepend to prompt before sending to GPT-5.1 Codex.
   - Soft limit enforcement: if final prompt > 120k, drop lowest-similarity chunk and retry.
   - Logging: each call records which summaries were used (for audit + future improvements).

4. **Automation Hooks**
   - GitHub Action / scheduled script runs `agent_context_distiller.py` nightly.
   - After summary refresh, push sanitized subset to `public-export/context/` for the public mirror.
   - Optional alert if summaries older than 48h.

## Implementation Order
1. Build summarizer agent + DuckDB/Qdrant persistence.
2. Add `gpt-scope` command referencing the summaries + file manifest.
3. Enhance Codex clients to consume retrieval pipeline.
4. Wire automation (cron/GitHub Action) to refresh summaries.

## Open Questions
- Should summaries include metrics snapshots (latency, success) or link to dashboards?
- How to handle private vs public summaries? (Proposal: `visibility` column.)
- Token estimation method (use `tiktoken` for accuracy?).

## Success Criteria
- Copilot/GPT requests no longer hit token limit.
- Operators can run `swarmctl gpt-scope --topic evolution` and receive <20k-token pack instantly.
- Codex responses cite retrieved summaries (storyhook + subsystem metadata).
