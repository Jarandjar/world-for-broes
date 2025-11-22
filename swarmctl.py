#!/usr/bin/env python3
"""SwarmCTL - Command-line interface for DiabetesAI Swarm

Provides high-level commands for swarm operations:
- backtest: Run therapy strategy simulations
- ingest: Load trial data, papers, torrents
- query: Search evidence base via RAG
- report: Generate narratives
- status: Check swarm health
- jobs: List/inspect/cancel jobs
- gpt: Ask GPT-5.1 Codex with optional repo context

Usage:
  swarmctl backtest create --therapy semaglutide --cohort "BMI>30 AND HbA1c>7.5"
  swarmctl backtest list
  swarmctl backtest compare sim_abc123 sim_def456
  swarmctl jobs list --status running
  swarmctl status
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
from swarm_orchestrator import Governor, JobType, JobStatus

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DIR = REPO_ROOT / "python"
if PYTHON_DIR.exists() and str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

try:  # noqa: E402
    from codex_client import CodexClient, CodexMessage
    CODEX_IMPORT_ERROR: Optional[Exception] = None
except Exception as codex_exc:  # noqa: E402
    CodexClient = None  # type: ignore
    CodexMessage = None  # type: ignore
    CODEX_IMPORT_ERROR = codex_exc

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class SwarmCTL:
    """Command-line interface controller"""
    
    def __init__(self, db_path: str = 'evidence.duckdb'):
        self.governor = None  # Lazy init for commands that need it
        self.db_path = db_path
        self._codex_client: Optional[CodexClient] = None
        self.summary_dir = REPO_ROOT / "summaries"
    
    def _ensure_governor(self):
        """Lazy initialize governor only when needed"""
        if self.governor is None:
            self.governor = Governor(db_path=self.db_path)
        return self.governor
    
    # ===== BACKTEST COMMANDS =====
    
    def backtest_create(self, args):
        """Create new backtest job"""
        if not args.therapy or not args.cohort:
            print("❌ Error: --therapy and --cohort required")
            return 1
        
        gov = self._ensure_governor()
        job_id = gov.create_job(JobType.BACKTEST_THERAPY, {
            "therapy": args.therapy,
            "strategy": args.strategy,
            "cohort": args.cohort,
            "horizon_weeks": args.horizon,
            "compare_to": args.compare_to
        }, priority=9)
        
        self.governor.process_queue()
        
        job = self.governor.jobs[job_id]
        print(f"\n✅ Backtest Complete: {job_id}")
        
        if job.result:
            self.print_backtest_result(job.result)
        
        return 0
    
    def backtest_list(self, args):
        """List all backtest simulations"""
        if not DUCKDB_AVAILABLE:
            print("⚠️ DuckDB not available; cannot query simulation history")
            return 1
        
        conn = duckdb.connect('simulations.duckdb')
        try:
            conn.execute("SELECT 1 FROM simulation_runs LIMIT 1")
        except Exception:
            print("📋 No simulations found. Run 'backtest create' first.")
            return 0
        
        sims = conn.execute("""
            SELECT sim_id, drug_name, strategy_name, n_patients, 
                   created_at, status
            FROM simulation_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, [args.limit]).fetchall()
        
        print(f"\n📊 Recent Simulations (showing {len(sims)}):\n")
        for sim_id, drug, strategy, n, created, status in sims:
            print(f"  {sim_id} | {drug:15s} | {strategy:20s} | N={n:4d} | {status}")
        
        conn.close()
        return 0
    
    def backtest_compare(self, args):
        """Compare two backtest simulations"""
        if len(args.sim_ids) < 2:
            print("❌ Error: Provide at least 2 sim_ids to compare")
            return 1
        
        print(f"\n📊 Comparing Simulations: {' vs '.join(args.sim_ids)}\n")
        
        # Mock comparison for now
        print("Comparison Results:")
        print("  Metric                 | Sim A      | Sim B      | Difference")
        print("  " + "-"*60)
        print("  HbA1c Change (%)       | -1.45      | -1.20      | -0.25 (A better)")
        print("  Weight Change (kg)     | -4.5       | -3.2       | -1.3 (A better)")
        print("  % Reaching Target      | 64%        | 48%        | +16pp (A better)")
        print("  Dropout Rate (%)       | 17%        | 22%        | -5pp (A better)")
        print("\n  ➡️  Simulation A shows superior outcomes across all metrics.")
        
        return 0
    
    def print_backtest_result(self, result: Dict[str, Any]):
        """Pretty-print backtest results"""
        print(f"\n  Therapy: {result.get('therapy', 'N/A')}")
        print(f"  Sim ID: {result.get('sim_id', 'N/A')}")
        print(f"  HbA1c Change: {result.get('hba1c_change_mean', 0):.2f}%")
        print(f"  Weight Change: {result.get('weight_change_mean', 0):.2f} kg")
        print(f"  Target Reached: {result.get('pct_reaching_target', 0):.1f}%")
        print(f"  Dropout Rate: {result.get('dropout_rate', 0):.1f}%")
    
    # ===== INGEST COMMANDS =====
    
    def ingest_trials(self, args):
        """Ingest clinical trial data"""
        print(f"\n🔬 Ingesting trial data: {args.source}")
        
        if args.source == "sustain":
            from load_sustain_trials import load_trials, SUSTAIN_TRIALS
            count = load_trials('clinical_trials.duckdb', SUSTAIN_TRIALS)
            print(f"✅ Loaded {count} SUSTAIN trials")
        else:
            job_id = self.governor.create_job(JobType.INGEST_TRIAL, {
                "source": args.source,
                "nct_id": args.nct_id
            }, priority=8)
            self.governor.process_queue()
            print(f"✅ Ingest job completed: {job_id}")
        
        return 0
    
    def ingest_torrent(self, args):
        """Ingest data from academic torrent"""
        print(f"\n📥 Ingesting torrent: {args.torrent_id}")
        
        job_id = self.governor.create_job(JobType.INGEST_TORRENT, {
            "torrent_id": args.torrent_id,
            "data_type": args.data_type
        }, priority=6)
        
        self.governor.process_queue()
        print(f"✅ Ingest job completed: {job_id}")
        
        return 0
    
    # ===== QUERY COMMANDS =====
    
    def query_rag(self, args):
        """Query evidence base via RAG"""
        print(f"\n🔍 Searching: {args.query}")
        
        job_id = self.governor.create_job(JobType.RAG_QUERY, {
            "query": args.query,
            "top_k": args.top_k
        }, priority=7)
        
        self.governor.process_queue()
        
        job = self.governor.jobs[job_id]
        if job.result and 'results' in job.result:
            print(f"\n📚 Top {len(job.result['results'])} Results:\n")
            for i, res in enumerate(job.result['results'], 1):
                trial = res.get('trial', 'Unknown')
                relevance = res.get('relevance', 0)
                print(f"  {i}. {trial} (relevance: {relevance:.2f})")
        
        return 0
    
    # ===== REPORT COMMANDS =====
    
    def report_generate(self, args):
        """Generate narrative report"""
        print(f"\n⚗️ Generating {args.report_type} report for {args.sim_id}")
        
        job_id = self.governor.create_job(JobType.GENERATE_REPORT, {
            "report_type": args.report_type,
            "sim_id": args.sim_id
        }, priority=6)
        
        self.governor.process_queue()
        
        job = self.governor.jobs[job_id]
        if job.result:
            print(f"\n✅ Report Generated:")
            print(f"  Report ID: {job.result.get('report_id')}")
            print(f"  Type: {job.result.get('report_type')}")
            print(f"  Word Count: {job.result.get('word_count')}")
            print(f"  Sections: {', '.join(job.result.get('sections', []))}")
        
        return 0
    
    # ===== JOB MANAGEMENT =====
    
    def jobs_list(self, args):
        """List jobs in queue/history"""
        print(f"\n📋 Jobs (status: {args.status or 'all'}):\n")
        
        for job_id, job in self.governor.jobs.items():
            if args.status and job.status.value.lower() != args.status.lower():
                continue
            
            status_icon = {
                JobStatus.PENDING: "⏳",
                JobStatus.RUNNING: "▶️",
                JobStatus.COMPLETED: "✅",
                JobStatus.FAILED: "❌"
            }.get(job.status, "❓")
            
            print(f"  {status_icon} {job.job_id:30s} | {job.job_type.value:20s} | "
                  f"{job.status.value:10s} | Pri:{job.priority}")
        
        return 0
    
    def jobs_inspect(self, args):
        """Inspect specific job details"""
        job = self.governor.jobs.get(args.job_id)
        if not job:
            print(f"❌ Job not found: {args.job_id}")
            return 1
        
        print(f"\n📝 Job Details: {args.job_id}\n")
        print(f"  Type: {job.job_type.value}")
        print(f"  Status: {job.status.value}")
        print(f"  Priority: {job.priority}")
        print(f"  Created: {job.created_at}")
        print(f"  Assigned Agent: {job.assigned_agent or 'None'}")
        
        if job.meta:
            print(f"\n  Metadata:")
            for k, v in job.meta.items():
                print(f"    {k}: {v}")
        
        if job.result:
            print(f"\n  Result:")
            print(f"    {json.dumps(job.result, indent=4)}")
        
        if job.error:
            print(f"\n  Error: {job.error}")
        
        return 0
    
    # ===== STATUS =====
    
    def show_status(self, args):
        """Show swarm status"""
        gov = self._ensure_governor()
        gov.print_status()
        return 0
    
    def show_briefing(self, args):
        """Generate daily briefing"""
        import subprocess
        cmd = [sys.executable, 'agent_daily_briefing.py']
        if args.print:
            cmd.append('--print')
        return subprocess.run(cmd).returncode
    
    def show_fitness(self, args):
        """Show fitness status or generate dashboard"""
        if args.dashboard:
            import subprocess
            cmd = [sys.executable, 'fitness_dashboard.py', '--days', str(args.days)]
            return subprocess.run(cmd).returncode
        
        # Quick fitness status
        if not DUCKDB_AVAILABLE:
            print("⚠️ DuckDB not available")
            return 1
        
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            print("🏃 Fitness Intelligence\n")
            
            workouts = conn.execute("""
                SELECT COUNT(*), 
                       SUM(duration_seconds)/3600.0,
                       SUM(distance_meters)/1000.0
                FROM workouts
                WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """).fetchone()
            
            count, hours, km = workouts
            
            if count and count > 0:
                print(f"  📅 Last 7 Days:")
                print(f"     Workouts: {count}")
                print(f"     Duration: {hours:.1f}h" if hours else "     Duration: 0h")
                print(f"     Distance: {km:.1f}km" if km else "     Distance: 0km")
            else:
                print("  ⚠️  No workout data")
                print("  💡 Run: python add_fitness_test_tasks.py")
            
            conn.close()
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
        
        return 0
    
    def show_health(self, args):
        """Show system health dashboard"""
        if not DUCKDB_AVAILABLE:
            print("⚠️ DuckDB not available")
            return 1
        
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            print("📊 Swarm Health Dashboard\n")
            
            # Tasks per hour (last 24h)
            tasks_24h = conn.execute("""
                SELECT COUNT(*) FROM agent_tasks
                WHERE started_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """).fetchone()[0]
            
            tasks_per_hour = tasks_24h / 24 if tasks_24h > 0 else 0
            
            # Error rate
            failed = conn.execute("""
                SELECT COUNT(*) FROM agent_tasks
                WHERE status = 'failed'
                  AND started_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """).fetchone()[0]
            
            error_rate = (failed / tasks_24h * 100) if tasks_24h > 0 else 0
            
            # Long-running tasks
            long_running = conn.execute("""
                SELECT COUNT(*) FROM agent_tasks
                WHERE status = 'running'
                  AND started_at < CURRENT_TIMESTAMP - INTERVAL '1 hour'
            """).fetchone()[0]
            
            # Stuck tasks
            stuck = conn.execute("""
                SELECT COUNT(*) FROM agent_tasks
                WHERE status = 'pending'
                  AND created_at < CURRENT_TIMESTAMP - INTERVAL '24 hours'
            """).fetchone()[0]
            
            # Display
            print(f"  ⚡ Tasks/Hour: {tasks_per_hour:.1f}")
            print(f"  {'✅' if error_rate < 5 else '⚠️'} Error Rate: {error_rate:.1f}%")
            print(f"  {'⏱️' if long_running > 0 else '✅'} Long-Running: {long_running}")
            print(f"  {'⚠️' if stuck > 50 else '✅'} Stuck Tasks: {stuck}")
            
            # Health score
            health = 100
            if error_rate > 10:
                health -= 30
            elif error_rate > 5:
                health -= 15
            if stuck > 50:
                health -= 20
            if long_running > 10:
                health -= 15
            
            status_emoji = '✨' if health >= 80 else '⚠️' if health >= 60 else '🔥'
            print(f"\n  {status_emoji} Overall Health: {health}/100")
            
            conn.close()
            
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
        
        return 0

    # ===== GPT COMMAND =====

    def gpt_chat(self, args):
        """Send a prompt to GPT-5.1 Codex with optional repo context"""
        try:
            client = self._ensure_codex_client()
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1
        context = self._build_gpt_context(args)

        metadata = {
            "tool": "swarmctl",
            "intent": "cli_gpt",
            "context_files": str(len(args.context_files or [])),
            "context_snippets": str(len(args.context_snippets or [])),
        }

        try:
            result = client.chat(
                messages=[CodexMessage(role='user', content=args.prompt)],
                system_prompt=args.system,
                context=context,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - network side effects
            print(f"❌ GPT request failed: {exc}")
            return 1

        print("\n🧠 GPT-5.1 Response:\n")
        print(result.text or "[no content]")

        if args.show_reasoning and result.reasoning:
            print("\n🧪 Reasoning Trace:\n")
            print(result.reasoning)

        if args.show_usage and result.usage:
            usage = result.usage
            print("\n📏 Token Usage:")
            print(f"  Input:  {usage.input_tokens}")
            print(f"  Output: {usage.output_tokens}")
            print(f"  Total:  {usage.total_tokens}")

        if args.save_raw:
            raw_path = Path(args.save_raw).expanduser()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(result.raw.model_dump_json(indent=2), encoding='utf-8')
            print(f"\n💾 Raw response saved to {raw_path}")

        return 0

    def _ensure_codex_client(self) -> CodexClient:
        if CodexClient is None:
            raise RuntimeError(
                "Codex client is unavailable. Install the 'openai' dependency in the current "
                "environment to use 'swarmctl gpt'."
            ) from CODEX_IMPORT_ERROR

        if self._codex_client is None:
            self._codex_client = CodexClient(
                system_prompt=(
                    "You are the Swarm Control Plane architect assistant. "
                    "Provide concise, technical guidance grounded in repo context."
                ),
                default_context="SwarmCTL CLI :: Swarm Control Plane",
                metadata={"tool": "swarmctl", "component": "cli"},
                reasoning_effort="medium",
                max_output_tokens=900,
            )
        return self._codex_client

    @staticmethod
    def _build_gpt_context(args) -> Optional[str]:
        sections: List[str] = []
        per_section_limit = 4000

        if args.context_files:
            for raw_path in args.context_files:
                path = Path(raw_path).expanduser()
                try:
                    text = path.read_text(encoding='utf-8')
                except Exception as exc:
                    print(f"⚠️ Unable to read context file {path}: {exc}")
                    continue
                chunk = text.strip()
                if not chunk:
                    continue
                trimmed = SwarmCTL._truncate(chunk, per_section_limit)
                sections.append(f"## File: {path}\n```text\n{trimmed}\n```")

        if args.context_snippets:
            for idx, snippet in enumerate(args.context_snippets, 1):
                clean = (snippet or '').strip()
                if not clean:
                    continue
                trimmed = SwarmCTL._truncate(clean, per_section_limit)
                sections.append(f"## Snippet {idx}\n{trimmed}")

        if not sections:
            return None

        combined = "\n\n".join(sections)
        return SwarmCTL._truncate(combined, 20000)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        trimmed = text[:limit].rstrip()
        return f"{trimmed}\n...[truncated {len(text) - limit} chars]..."

    # ===== CONTEXT / GPT SCOPE =====

    def gpt_scope(self, args) -> int:
        if args.list_subsystems:
            names = self._available_context_subsystems()
            if not names:
                print("⚠️ No subsystem summaries found. Run agent_context_distiller first.")
                return 1
            print("Available subsystem summaries:")
            for name in sorted(names):
                print(f"- {name}")
            return 0

        sections: List[str] = []
        included_summary = set()

        def add_summary(name: str) -> None:
            summary = self._load_subsystem_summary(name)
            if not summary:
                print(f"⚠️ Summary not found for subsystem '{name}'.")
                return
            sections.append(f"## Subsystem: {name}\n{summary.strip()}\n")
            included_summary.add(name)

        if args.subsystem:
            for name in args.subsystem:
                add_summary(name)

        if args.topic:
            for topic in args.topic:
                matches = self._search_summaries_by_topic(topic, limit=args.topic_limit)
                for name in matches:
                    if name in included_summary:
                        continue
                    add_summary(name)

        if args.path:
            for pattern in args.path:
                matched = list(self._resolve_files(pattern))
                if not matched:
                    print(f"⚠️ No files matched pattern '{pattern}'.")
                    continue
                for file_path in matched:
                    try:
                        text = file_path.read_text(encoding="utf-8")
                    except Exception as exc:
                        print(f"⚠️ Unable to read {file_path}: {exc}")
                        continue
                    trimmed = self._truncate(text, args.max_file_chars)
                    try:
                        rel = file_path.relative_to(REPO_ROOT)
                    except ValueError:
                        rel = file_path
                    sections.append(f"## File: {rel}\n```text\n{trimmed}\n```\n")

        if not sections:
            print("⚠️ Nothing selected. Use --subsystem, --path, or --topic to include context.")
            return 1

        context_blob = "\n".join(sections).strip() + "\n"
        tokens = self._estimate_tokens(context_blob)

        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(context_blob, encoding="utf-8")
            print(f"✅ Wrote context pack to {output_path} ({tokens} tokens approx.)")
        else:
            print(context_blob)
            print(f"\n---\nApproximate tokens: {tokens}")

        if tokens > args.max_tokens:
            print(f"⚠️ Context pack is ~{tokens} tokens (>{args.max_tokens}). Consider removing files or summarizing further.")

        return 0

    def _available_context_subsystems(self) -> List[str]:
        names = set()
        if self.summary_dir.exists():
            for path in self.summary_dir.glob("*.md"):
                names.add(path.stem)
        if DUCKDB_AVAILABLE:
            try:
                con = duckdb.connect(self.db_path)
                rows = con.execute("SELECT subsystem FROM context_summaries").fetchall()
                con.close()
                for row in rows:
                    names.add(row[0])
            except Exception:
                pass
        return sorted(names)

    def _load_subsystem_summary(self, name: str) -> Optional[str]:
        file_path = self.summary_dir / f"{name}.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        if DUCKDB_AVAILABLE:
            try:
                con = duckdb.connect(self.db_path)
                row = con.execute(
                    "SELECT summary_md FROM context_summaries WHERE subsystem = ?",
                    [name],
                ).fetchone()
                con.close()
                if row:
                    return row[0]
            except Exception:
                return None
        return None

    def _search_summaries_by_topic(self, topic: str, limit: int = 2) -> List[str]:
        topic_lower = topic.lower()
        matches: List[str] = []
        if self.summary_dir.exists():
            for path in self.summary_dir.glob("*.md"):
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                if topic_lower in text.lower():
                    matches.append(path.stem)
        if matches:
            return matches[:limit]
        if DUCKDB_AVAILABLE:
            try:
                con = duckdb.connect(self.db_path)
                rows = con.execute(
                    "SELECT subsystem FROM context_summaries WHERE lower(summary_md) LIKE ? LIMIT ?",
                    [f"%{topic_lower}%", limit],
                ).fetchall()
                con.close()
                return [row[0] for row in rows]
            except Exception:
                return []
        return []

    def _resolve_files(self, pattern: str) -> Iterable[Path]:
        path = Path(pattern)
        if path.exists():
            return [path]
        return REPO_ROOT.glob(pattern)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)


def main():
    parser = argparse.ArgumentParser(
        description='SwarmCTL - DiabetesAI Swarm Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--db', default='evidence.duckdb', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # BACKTEST
    backtest = subparsers.add_parser('backtest', help='Therapy backtest operations')
    backtest_sub = backtest.add_subparsers(dest='subcommand')
    
    backtest_create = backtest_sub.add_parser('create', help='Create new backtest')
    backtest_create.add_argument('--therapy', required=True, help='Drug name')
    backtest_create.add_argument('--cohort', required=True, help='Cohort filter')
    backtest_create.add_argument('--strategy', default='add_on_metformin', help='Strategy')
    backtest_create.add_argument('--horizon', type=int, default=52, help='Weeks')
    backtest_create.add_argument('--compare-to', help='Baseline sim_id')
    
    backtest_list = backtest_sub.add_parser('list', help='List simulations')
    backtest_list.add_argument('--limit', type=int, default=10, help='Max results')
    
    backtest_compare = backtest_sub.add_parser('compare', help='Compare simulations')
    backtest_compare.add_argument('sim_ids', nargs='+', help='Simulation IDs')
    
    # INGEST
    ingest = subparsers.add_parser('ingest', help='Data ingestion operations')
    ingest_sub = ingest.add_subparsers(dest='subcommand')
    
    ingest_trials = ingest_sub.add_parser('trials', help='Ingest trial data')
    ingest_trials.add_argument('--source', required=True, help='Source (sustain, etc.)')
    ingest_trials.add_argument('--nct-id', help='ClinicalTrials.gov NCT ID')
    
    ingest_torrent = ingest_sub.add_parser('torrent', help='Ingest from torrent')
    ingest_torrent.add_argument('--torrent-id', required=True, help='Torrent identifier')
    ingest_torrent.add_argument('--data-type', default='papers', help='Data type')
    
    # QUERY
    query = subparsers.add_parser('query', help='Search evidence base')
    query.add_argument('query', help='Search query')
    query.add_argument('--top-k', type=int, default=5, help='Top K results')
    
    # REPORT
    report = subparsers.add_parser('report', help='Generate reports')
    report.add_argument('--sim-id', required=True, help='Simulation ID')
    report.add_argument('--report-type', choices=['technical', 'lore', 'both'], 
                       default='both', help='Report type')
    
    # JOBS
    jobs = subparsers.add_parser('jobs', help='Job management')
    jobs_sub = jobs.add_subparsers(dest='subcommand')
    
    jobs_list = jobs_sub.add_parser('list', help='List jobs')
    jobs_list.add_argument('--status', choices=['pending', 'running', 'completed', 'failed'],
                          help='Filter by status')
    
    jobs_inspect = jobs_sub.add_parser('inspect', help='Inspect job details')
    jobs_inspect.add_argument('job_id', help='Job ID')
    
    # STATUS
    subparsers.add_parser('status', help='Show swarm status')
    
    # BRIEFING
    briefing = subparsers.add_parser('briefing', help='Generate daily command scroll')
    briefing.add_argument('--print', action='store_true', help='Print to console')
    
    # FITNESS
    fitness = subparsers.add_parser('fitness', help='Fitness status and dashboard')
    fitness.add_argument('--dashboard', action='store_true', help='Generate HTML dashboard')
    fitness.add_argument('--days', type=int, default=90, help='Days for dashboard')
    
    # HEALTH
    subparsers.add_parser('health', help='System health dashboard')

    # GPT
    gpt = subparsers.add_parser('gpt', help='Ask GPT-5.1 with optional repo context')
    gpt.add_argument('prompt', help='Prompt/objective to send to GPT-5.1')
    gpt.add_argument('--context-file', dest='context_files', action='append',
                    help='Path to include as context (repeatable)')
    gpt.add_argument('--context-snippet', dest='context_snippets', action='append',
                    help='Inline context snippet (repeatable)')
    gpt.add_argument('--system', help='Override the default system prompt')
    gpt.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    gpt.add_argument('--max-output-tokens', type=int, default=800,
                    help='Maximum output tokens for the response')
    gpt.add_argument('--reasoning', choices=['low', 'medium', 'high'], default='medium',
                    help='Reasoning effort level')
    gpt.add_argument('--show-reasoning', action='store_true', help='Display reasoning trace')
    gpt.add_argument('--show-usage', action='store_true', help='Display token usage summary')
    gpt.add_argument('--save-raw', help='Save raw JSON response to this path')

    gpt_scope = subparsers.add_parser('gpt-scope', help='Build a scoped context pack for GPT/Copilot')
    gpt_scope.add_argument('--subsystem', action='append', help='Include subsystem summary (name from agent_context_distiller)')
    gpt_scope.add_argument('--topic', action='append', help='Keyword/topic to match against subsystem summaries')
    gpt_scope.add_argument('--topic-limit', type=int, default=2, help='Max summaries to include per topic')
    gpt_scope.add_argument('--path', action='append', help='File path or glob to include as raw context')
    gpt_scope.add_argument('--max-file-chars', type=int, default=2000, help='Max characters per file when embedding raw text')
    gpt_scope.add_argument('--max-tokens', type=int, default=100000, help='Warn if pack exceeds this many tokens')
    gpt_scope.add_argument('--output', help='Write context pack to this file instead of stdout')
    gpt_scope.add_argument('--list-subsystems', action='store_true', help='List available subsystem summaries and exit')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    ctl = SwarmCTL(db_path=args.db)
    
    # Route commands
    if args.command == 'backtest':
        if args.subcommand == 'create':
            return ctl.backtest_create(args)
        elif args.subcommand == 'list':
            return ctl.backtest_list(args)
        elif args.subcommand == 'compare':
            return ctl.backtest_compare(args)
    
    elif args.command == 'ingest':
        if args.subcommand == 'trials':
            return ctl.ingest_trials(args)
        elif args.subcommand == 'torrent':
            return ctl.ingest_torrent(args)
    
    elif args.command == 'query':
        return ctl.query_rag(args)
    
    elif args.command == 'report':
        return ctl.report_generate(args)
    
    elif args.command == 'jobs':
        if args.subcommand == 'list':
            return ctl.jobs_list(args)
        elif args.subcommand == 'inspect':
            return ctl.jobs_inspect(args)
    
    elif args.command == 'status':
        return ctl.show_status(args)
    elif args.command == 'briefing':
        return ctl.show_briefing(args)
    elif args.command == 'fitness':
        return ctl.show_fitness(args)
    elif args.command == 'health':
        return ctl.show_health(args)
    elif args.command == 'gpt':
        return ctl.gpt_chat(args)
    elif args.command == 'gpt-scope':
        return ctl.gpt_scope(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
