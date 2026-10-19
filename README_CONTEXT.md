# World for Broes

Context compression toolkit for AI agents: retrieval-augmented Codex client, subsystem summarizer, and scoped CLI to prevent token overflow in large codebases.

## Features

- **ContextRetriever**: Hybrid semantic search (Qdrant + DuckDB + file fallback) for subsystem summaries
- **CodexClient**: GPT-5.1 wrapper with automatic context injection via `context_query` parameter
- **agent_context_distiller**: Generate markdown summaries from subsystem files
- **swarmctl gpt-scope**: CLI command to assemble scoped context packs for GPT queries

## Installation

```bash
# Clone the repo
git clone https://github.com/Jarandjar/world-for-broes.git
cd world-for-broes

# Install dependencies
pip install -r requirements.txt

# Configure API keys
export OPENAI_API_KEY="your-key-here"
export EVIDENCE_DUCKDB_PATH="evidence.duckdb"  # optional

# Start Qdrant (optional, for vector search)
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Quick Start

### 1. Generate Subsystem Summaries

```bash
python agent_context_distiller.py governor --max-files 5
python agent_context_distiller.py --list
```

Summaries are saved to `summaries/` and indexed in DuckDB + Qdrant.

### 2. Query Scoped Context

```bash
python swarmctl.py gpt-scope --subsystem governor --max-tokens 2000
python swarmctl.py gpt-scope --topic "agent evolution" --max-tokens 4000
```

### 3. Use Retrieval-Augmented Codex Client

```python
from python.codex_client import CodexClient, CodexMessage

client = CodexClient(enable_retrieval=True)

result = client.chat(
    messages=[CodexMessage(role="user", content="How does agent evolution work?")],
    context_query="agent evolution and promotion logic",  # retrieves relevant summaries
    context_max_chars=4000
)

print(result.text)
```

## Architecture

```
world-for-broes/
├── python/
│   └── codex_client.py        # ContextRetriever + CodexClient
├── agent_context_distiller.py # Subsystem summarizer
├── swarmctl.py                # CLI with gpt-scope command
├── swarm_vector_db.py         # Qdrant integration
├── agent_sdk.py               # Event logging utilities
├── summaries/                 # Generated subsystem summaries
└── CONTEXT_COMPRESSION_PLAN.md
```

## How It Works

1. **Summarization**: `agent_context_distiller` crawls subsystems, produces ~1k-token summaries, stores in DuckDB + Qdrant
2. **Retrieval**: `ContextRetriever` performs semantic search across summaries when you provide a `context_query`
3. **Injection**: `CodexClient` prepends retrieved context to GPT prompts, staying under token limits

## Configuration

Environment variables:

- `OPENAI_API_KEY` – Required for GPT-5.1 access
- `EVIDENCE_DUCKDB_PATH` – Path to DuckDB database (default: `evidence.duckdb`)
- `CONTEXT_SUMMARY_DIR` – Summary output directory (default: `summaries/`)
- `QDRANT_HOST` – Qdrant server host (default: `localhost`)
- `QDRANT_PORT` – Qdrant server port (default: `6333`)

## Examples

### Generate All Subsystem Summaries

```bash
for subsystem in governor oracle harvester alchemist guardian; do
    python agent_context_distiller.py "$subsystem" --max-files 10 --max-chars 3000
done
```

### Query with Token Budget

```bash
python swarmctl.py gpt-scope --subsystem governor --max-tokens 1500 --format json
```

### Custom Retriever

```python
from python.codex_client import ContextRetriever

retriever = ContextRetriever(
    summary_dir="custom_summaries/",
    db_path="custom.duckdb"
)

context = retriever.retrieve("boss battles", top_k=5, max_chars=3000)
print(context)
```

## Contributing

Pull requests welcome. For major changes, open an issue first.

## Roadmap-to-cloud coding

Add coding ideas to [`ROADMAP.md`](ROADMAP.md) with an ID, rationale, expected outcome, and testable acceptance criteria. Ideas remain inert while marked `proposed`. After you deliberately change an item to `ready` and merge that change to `main`, the `Roadmap cloud coding` workflow creates a GitHub issue and starts a Copilot cloud task. The task is asked to implement only that item, run relevant tests, and open a pull request; human review and merge are still required.

Before enabling dispatch:

1. Enable Copilot cloud agent for this repository and confirm the token owner has a Copilot Business or Enterprise plan.
2. Create a fine-grained personal access token for the repository with **Agent tasks: read and write**, and save it as the Actions secret `COPILOT_AGENT_TOKEN`. The workflow uses the built-in `GITHUB_TOKEN` for issue creation.
3. Confirm Actions are enabled. The workflow only runs on `main`; it can be triggered by a `ROADMAP.md` push or manually from the Actions tab.

The cloud-agent task API is in public preview and requires a user-to-server token; installation tokens and the built-in `GITHUB_TOKEN` cannot start tasks. After the issue is launched, change its roadmap status to `in-progress`; change it to `done` after review and merge. If a run is interrupted after its issue receives a pending-launch marker, inspect the [cloud task list](https://docs.github.com/en/rest/agent-tasks/agent-tasks) before clearing the marker or retrying, to avoid duplicate work. Check active and completed cloud tasks with:

```bash
COPILOT_AGENT_TOKEN=... GITHUB_REPOSITORY=owner/repo python roadmap_agent.py tasks
python roadmap_agent.py validate
```

`swarmctl jobs list` continues to report the local swarm queue; GitHub cloud tasks have their own states and are listed separately by `roadmap_agent.py tasks`.

The existing swarm queue handles ingestion (`INGEST_TRIAL`, `INGEST_TORRENT`), retrieval and analysis (`RAG_QUERY`, `BACKTEST_THERAPY`, `STATS_ANALYSIS`), reporting and safety (`GENERATE_REPORT`, `SAFETY_CHECK`, `HOURLY_STATS`), and fitness (`FITNESS_SYNC`, `FITNESS_ANALYSIS`). Its lifecycle states are pending, assigned, running, completed, and failed. The current coder task mappings (`code_todo`, `code_fixme`, `code_hack`, and `code_xxx`) map to `HOURLY_STATS`; they identify code markers but do not implement them. For that reason, roadmap-driven code work is dispatched to the separate Copilot cloud task API rather than being reported as a local swarm job.

See GitHub's [Copilot cloud agent API guide](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/use-cloud-agent-via-the-api) for supported task states, permissions, and API details.

## License

MIT

## Credits

Built for the Swarm Control Plane project. See [CONTEXT_COMPRESSION_PLAN.md](CONTEXT_COMPRESSION_PLAN.md) for design details.
