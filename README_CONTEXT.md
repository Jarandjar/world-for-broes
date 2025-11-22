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

## License

MIT

## Credits

Built for the Swarm Control Plane project. See [CONTEXT_COMPRESSION_PLAN.md](CONTEXT_COMPRESSION_PLAN.md) for design details.
