"""Context distillation agent for the Swarm.

Generates markdown summaries for each major subsystem, writes them to
`summaries/<subsystem>.md`, stores them in DuckDB, and uploads embeddings to
Qdrant for retrieval-aware prompting.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import duckdb

try:
    from codex_client import CodexClient, CodexMessage
except ImportError:  # pragma: no cover - optional during bootstrap
    CodexClient = None  # type: ignore
    CodexMessage = None  # type: ignore

from swarm_vector_db import SwarmVectorDB

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "summaries"
DEFAULT_DB_PATH = os.getenv("CONTEXT_SUMMARY_DUCKDB_PATH") or os.getenv("EVIDENCE_DUCKDB_PATH", "evidence.duckdb")


@dataclass
class Subsystem:
    name: str
    description: str
    paths: Sequence[str]
    docs: Sequence[str]


SUBSYSTEMS: Dict[str, Subsystem] = {
    "governor": Subsystem(
        name="governor",
        description="Express/TypeScript control plane for REST APIs, Demiurge hooks, and diagram routes",
        paths=(
            "swarm-evolution/packages/governor/src/**/*.ts",
            "swarm-evolution/packages/governor/src/**/*.tsx",
            "swarmctl.py",
        ),
        docs=("SWARM_LIVE_OPS_README.md", "SWARM_OPERATIONS_GUIDE.md"),
    ),
    "demiurge": Subsystem(
        name="demiurge",
        description="GPT-5.1 meta-governor orchestration plus prompts and edict sinks",
        paths=(
            "swarm-evolution/packages/governor/src/services/Demiurge*.ts",
            "demiurge_prompt.md",
            "python/demiurge_cycle_runner.py",
        ),
        docs=("SWARM_CONSCIOUSNESS_REPORT.md", "SWARM_LIVING_REPORT.md"),
    ),
    "evolution": Subsystem(
        name="evolution",
        description="Hatchling spawning, tournaments, ranking, genome math, and XP tracking",
        paths=(
            "agent_evolution.py",
            "global_evolution_engine.py",
            "swarm_evolution.py",
            "swarm-evolution/**/*.ts",
        ),
        docs=("SWARM_EVOLUTION_README.md", "SWARM_SPECIES_TAXONOMY.md"),
    ),
    "bosses": Subsystem(
        name="bosses",
        description="Chaos engineering via boss encounters, loot tables, and mythology telemetry",
        paths=(
            "agent_boss_battles.py",
            "mythology_telemetry.py",
            "mythology_telemetry.duckdb",
        ),
        docs=("SWARM_MYTHOLOGY_INDEX.md", "SWARM_RITUAL_CALENDAR.md"),
    ),
    "quests": Subsystem(
        name="quests",
        description="Technical debt quests triggered by audits, regressions, and coverage drops",
        paths=("agent_quests.py", "agent_auditor.py", "agent_auditlogger.py"),
        docs=("QUEST_SYSTEM_README.md", "SWARM_AUTONOMY_STATUS.md"),
    ),
    "lore": Subsystem(
        name="lore",
        description="Chronicle generation, codex upkeep, consciousness reports, and lore ingestion",
        paths=(
            "agent_chronicle.py",
            "SWARM_CODEX.md",
            "SWARM_CONSCIOUSNESS_REPORT.md",
            "SWARM_LIVING_REPORT.md",
        ),
        docs=("SWARM_CODEX_ENCYCLOPEDIA.md", "SWARM_MYTHOLOGY_COMPLETE.md"),
    ),
    "cli": Subsystem(
        name="cli",
        description="`swarmctl` commands, GPT helpers, and live ops tooling",
        paths=("swarmctl.py", "swarmctl_*.py", "SWARMCTL_CHEATSHEET.md"),
        docs=("SWARM_QUICK_COMMANDS.md", "SWARM_LIVE_OPS_README.md"),
    ),
    "docs": Subsystem(
        name="docs",
        description="Living documentation, launch checklists, runbooks, public export tooling",
        paths=(
            "docs/**/*.md",
            "LAUNCH_CHECKLIST.md",
            "RUNBOOK_SWARM_ONLINE.md",
            "public-export/**/*.md",
        ),
        docs=("SWARM_LIVING_REPORT.md", "public-export/README.md"),
    ),
    "gpu": Subsystem(
        name="gpu",
        description="GPU embedding helper, vector DB utilities, and retrieval plumbing",
        paths=("gpu_embeddings.py", "swarm_vector_db.py", "swarm_vector_db.py"),
        docs=("GPU_EMBEDDINGS_README.md", "SWARM_VECTOR_GUIDE.md"),
    ),
    "observability": Subsystem(
        name="observability",
        description="Live ops loop, resource monitors, latency dashboards, anomaly detectors",
        paths=("swarm_live_ops.py", "swarm_resource_monitor.py", "SWARM_STATUS.md"),
        docs=("SWARM_STATUS_REPORT.md", "SWARM_LIVE_UPDATE_GUIDE.md"),
    ),
}


def list_subsystems() -> None:
    print("Available subsystems:")
    for name, subsystem in SUBSYSTEMS.items():
        print(f"- {name:<14} {subsystem.description}")


def resolve_files(patterns: Sequence[str], limit: int) -> List[Path]:
    seen: List[Path] = []
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file():
                seen.append(path)
                if len(seen) >= limit:
                    return seen
    return seen


def read_snippets(files: Sequence[Path], max_chars: int, per_file: int) -> str:
    total = 0
    chunks: List[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        snippet = text[:per_file]
        chunks.append(f"### File: {path.relative_to(REPO_ROOT)}\n{snippet}\n")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(chunks)


def build_prompt(subsystem: Subsystem, context_blob: str) -> str:
    doc_list = "\n".join(f"- {doc}" for doc in subsystem.docs)
    template = f"""
Summarize the SWARM subsystem "{subsystem.name}".
Role description: {subsystem.description}
Referenced docs:\n{doc_list or '- (none)'}
Use the following markdown format and keep the entire response under ~1k tokens:

#### {subsystem.name.title()} Subsystem

**Scope:** <1-2 sentences about responsibilities and directories>
**Key APIs/classes:**
- ...
**Current status / TODO:**
- ...
**Related docs:**
- ...

Context excerpts:
{context_blob or '(no files provided)'}
"""
    return template.strip()


def summarize_with_llm(prompt: str) -> str:
    if not CodexClient:
        raise RuntimeError("codex_client is unavailable; cannot summarize with GPT")
    client = CodexClient(system_prompt="You are the official SWARM technical archivist.")
    result = client.chat(messages=[CodexMessage(role="user", content=prompt)])
    text = result.text.strip()
    if not text:
        raise RuntimeError("Codex returned an empty summary")
    return text


def fallback_summary(subsystem: Subsystem, files: Sequence[Path]) -> str:
    file_list = "\n".join(f"- {p.relative_to(REPO_ROOT)}" for p in files) or "- (no files captured)"
    doc_list = "\n".join(f"- {doc}" for doc in subsystem.docs) or "- (none)"
    return f"""#### {subsystem.name.title()} Subsystem

**Scope:** {subsystem.description}
**Key APIs/classes:**
{file_list}
**Current status / TODO:**
- Summaries require OPENAI_API_KEY to generate detailed context.
**Related docs:**
{doc_list}
"""


def estimate_tokens(text: str) -> int:
    try:  # pragma: no cover - optional dep
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def ensure_duckdb_table(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS context_summaries (
            subsystem TEXT PRIMARY KEY,
            summary_md TEXT,
            tokens INTEGER,
            refreshed_at TIMESTAMP
        )
        """
    )
    return con


def upsert_duckdb(con: duckdb.DuckDBPyConnection, subsystem: str, summary: str, tokens: int, refreshed_at: datetime) -> None:
    con.execute(
        "INSERT OR REPLACE INTO context_summaries VALUES (?, ?, ?, ?)",
        [subsystem, summary, tokens, refreshed_at],
    )


def upsert_vector(subsystem: str, summary: str, tokens: int, refreshed_at: datetime) -> None:
    vdb = SwarmVectorDB()
    if not vdb.client:
        return
    if not vdb.ensure_collection("context_summaries"):
        return
    from qdrant_client.models import PointStruct
    import uuid

    vector = vdb.embed_text([summary])[0]
    point = PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"context::{subsystem}")),
        vector=vector,
        payload={
            "subsystem": subsystem,
            "tokens": tokens,
            "refreshed_at": refreshed_at.isoformat(),
            "collection": "context_summaries",
            "content": summary,
        },
    )
    vdb.client.upsert(collection_name="context_summaries", points=[point])


def write_output_file(name: str, summary: str, refreshed_at: datetime) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.md"
    meta = f"\n_Last refreshed: {refreshed_at.isoformat()}_\n"
    path.write_text(f"{summary.strip()}\n{meta}\n", encoding="utf-8")
    return path


def generate_summary(subsystem: Subsystem, *, max_files: int, max_chars: int, per_file: int, use_llm: bool) -> Dict:
    files = resolve_files(subsystem.paths, limit=max_files)
    context_blob = read_snippets(files, max_chars=max_chars, per_file=per_file)
    summary_md: str
    if use_llm:
        try:
            summary_md = summarize_with_llm(build_prompt(subsystem, context_blob))
        except Exception as exc:
            print(f"⚠️  LLM summarization failed for {subsystem.name}: {exc}. Falling back to heuristic summary.")
            summary_md = fallback_summary(subsystem, files)
    else:
        summary_md = fallback_summary(subsystem, files)
    tokens = estimate_tokens(summary_md)
    refreshed_at = datetime.now(timezone.utc)
    return {
        "subsystem": subsystem.name,
        "summary": summary_md,
        "tokens": tokens,
        "refreshed_at": refreshed_at,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SWARM subsystem summaries")
    parser.add_argument(
        "subsystems",
        nargs="*",
        help="Subsystems to summarize (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available subsystems")
    parser.add_argument("--max-files", type=int, default=12, help="Max files per subsystem")
    parser.add_argument("--max-chars", type=int, default=8000, help="Max total characters per subsystem")
    parser.add_argument("--per-file", type=int, default=1500, help="Max characters per file excerpt")
    parser.add_argument("--no-llm", action="store_true", help="Skip Codex and use heuristic summaries")
    parser.add_argument("--skip-db", action="store_true", help="Do not write to DuckDB")
    parser.add_argument("--skip-vector", action="store_true", help="Do not upsert into Qdrant")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="DuckDB path for context_summaries table")
    args = parser.parse_args()

    if args.list:
        list_subsystems()
        return

    if args.subsystems:
        unknown = [name for name in args.subsystems if name not in SUBSYSTEMS]
        if unknown:
            parser.error(f"Unknown subsystem(s): {', '.join(unknown)}")
    targets = args.subsystems or list(SUBSYSTEMS.keys())
    use_llm = not args.no_llm

    db_con = None
    if not args.skip_db:
        db_con = ensure_duckdb_table(args.db_path)

    for name in targets:
        subsystem = SUBSYSTEMS[name]
        print(f"📘 Summarizing {name}...")
        data = generate_summary(
            subsystem,
            max_files=args.max_files,
            max_chars=args.max_chars,
            per_file=args.per_file,
            use_llm=use_llm,
        )
        output_path = write_output_file(name, data["summary"], data["refreshed_at"])
        print(f"   → wrote {output_path.relative_to(REPO_ROOT)} ({data['tokens']} tokens)")

        if db_con:
            upsert_duckdb(db_con, data["subsystem"], data["summary"], data["tokens"], data["refreshed_at"])
            print("   → stored summary in DuckDB")

        if not args.skip_vector:
            try:
                upsert_vector(data["subsystem"], data["summary"], data["tokens"], data["refreshed_at"])
                print("   → upserted embedding in Qdrant")
            except Exception as exc:
                print(f"   ⚠️  Qdrant upsert failed: {exc}")

    if db_con:
        db_con.close()


if __name__ == "__main__":
    main()
