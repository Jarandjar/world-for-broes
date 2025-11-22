#!/usr/bin/env python3
"""
SWARM LIVE OPS MODE - Continuous Autonomous Operation

This is the "Run Until I Say Stop" mode for the DiabetesAI Swarm.

Continuously:
- Monitors swarm status and agent health
- Auto-enqueues tasks from agent_tasks table
- Processes job queue
- Checks logs and metrics
- Detects failures, slow agents, and anomalies
- Proposes fixes and evolution experiments
- Logs everything for lore generation

Usage:
    python swarm_live_ops.py              # Start continuous mode
    python swarm_live_ops.py --interval 5  # Check every 5 seconds
    python swarm_live_ops.py --once        # Single cycle only
    
Stop with CTRL+C
"""
from __future__ import annotations

import argparse
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import traceback

# Import swarm components
from swarm_orchestrator import Governor, JobType, JobStatus
from agent_ranking import AgentRanking
from agent_metrics import AgentMetrics
from agent_task_discovery import TaskDiscoveryAgent
from swarm_output_formatters import get_formatter

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("⚠️  Warning: DuckDB not available. Some features disabled.")


class SwarmLiveOps:
    """Continuous swarm monitoring and operation"""
    
    def __init__(self, interval: int = 10, db_path: str = 'evidence.duckdb', output_style: str = 'compact'):
        self.interval = interval
        self.governor = Governor(db_path=db_path)
        self.db_path = db_path
        self.cycle_count = 0
        self.start_time = datetime.now()
        self.formatter = get_formatter(output_style)
        self.output_style = output_style
        
        # Task discovery agent
        self.task_discovery = TaskDiscoveryAgent(db_path=db_path)
        self.last_discovery_cycle = datetime.now()
        
        # Tracking
        self.last_errors: List[str] = []
        self.slow_agents: Dict[str, int] = {}
        self.degraded_agents: set = set()
        self.total_jobs_processed = 0
        self.evolution_suggestions: List[str] = []
        
        # Create logs directory
        self.log_dir = Path("logs/swarm")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        if DUCKDB_AVAILABLE:
            self.init_metrics_db()
    
    def init_metrics_db(self):
        """Initialize metrics tracking"""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swarm_live_metrics (
                timestamp TIMESTAMP,
                cycle_number INTEGER,
                jobs_pending INTEGER,
                jobs_running INTEGER,
                jobs_completed INTEGER,
                jobs_failed INTEGER,
                agents_busy INTEGER,
                agents_idle INTEGER,
                queue_backlog INTEGER,
                avg_latency_sec DOUBLE,
                errors_detected INTEGER,
                evolution_proposals INTEGER
            )
        """)
        conn.close()
    
    def auto_enqueue_tasks(self, limit: int = 50) -> int:
        """Pull tasks from agent_tasks table and create Governor jobs
        
        Args:
            limit: Max tasks to enqueue per cycle (default 50 for max CPU utilization)
        """
        if not DUCKDB_AVAILABLE:
            return 0
        
        conn = duckdb.connect(self.db_path)
        
        # Check if agent_tasks exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if 'agent_tasks' not in table_names:
            conn.close()
            return 0
        
        # Get pending tasks sorted by priority (increased limit for max utilization)
        pending = conn.execute(f"""
            SELECT task_id, agent_name, task_type, task_data, priority
            FROM agent_tasks
            WHERE status = 'pending'
            ORDER BY priority DESC, created_at ASC
            LIMIT {limit}
        """).fetchall()
        
        enqueued = 0
        for task_id, agent, task_type, task_data_json, priority in pending:
            try:
                task_data = json.loads(task_data_json) if task_data_json else {}
                
                # Map agent tasks to JobType
                job_type = self._map_task_to_job_type(agent, task_type)
                
                if job_type:
                    job_id = self.governor.create_job(job_type, task_data, priority=priority)
                    
                    # Update agent_tasks
                    conn.execute("""
                        UPDATE agent_tasks 
                        SET status = 'assigned', started_at = CURRENT_TIMESTAMP
                        WHERE task_id = ?
                    """, [task_id])
                    
                    enqueued += 1
            
            except Exception as e:
                print(f"⚠️  Error enqueuing task {task_id}: {e}")
        
        conn.close()
        return enqueued
    
    def _map_task_to_job_type(self, agent: str, task_type: str) -> JobType | None:
        """Map agent task types to Governor JobType enum - EXPANDED"""
        # Direct mappings for known combinations
        mapping = {
            # RAG agent tasks
            ('rag', 'index_evidence'): JobType.RAG_QUERY,
            ('rag', 'health_check'): JobType.HOURLY_STATS,
            ('rag', 'index_refresh'): JobType.RAG_QUERY,
            ('rag', 'cache_cleanup'): JobType.HOURLY_STATS,
            ('rag', 'rag_task'): JobType.RAG_QUERY,
            
            # Analytics agent tasks
            ('analytics', 'run_backtest'): JobType.BACKTEST_THERAPY,
            ('analytics', 'run_analytics'): JobType.STATS_ANALYSIS,
            ('analytics', 'hte_analysis'): JobType.STATS_ANALYSIS,
            ('analytics', 'metrics_report'): JobType.GENERATE_REPORT,
            ('analytics', 'analysis_task'): JobType.STATS_ANALYSIS,
            
            # Ingestion agent tasks
            ('ingestion', 'scan_raw_data'): JobType.INGEST_TRIAL,
            ('ingestion', 'extract_metadata'): JobType.INGEST_TRIAL,
            ('ingestion', 'scan_sources'): JobType.INGEST_TRIAL,
            ('ingestion', 'validate_data'): JobType.INGEST_TRIAL,
            ('ingestion', 'update_manifest'): JobType.INGEST_TRIAL,
            ('ingestion', 'ingestion_task'): JobType.INGEST_TRIAL,
            
            # Enhancement agent tasks
            ('enhancement', 'find_similar_trials'): JobType.RAG_QUERY,
            ('enhancement', 'find_improvements'): JobType.HOURLY_STATS,
            ('enhancement', 'tech_debt_scan'): JobType.HOURLY_STATS,
            ('enhancement', 'dependency_audit'): JobType.HOURLY_STATS,
            ('enhancement', 'log_rotation'): JobType.HOURLY_STATS,
            ('enhancement', 'cache_cleanup'): JobType.HOURLY_STATS,
            ('enhancement', 'general_task'): JobType.HOURLY_STATS,
            ('enhancement', 'master_todo'): JobType.HOURLY_STATS,
            
            # Performance agent tasks
            ('performance', 'evaluate_strategy'): JobType.BACKTEST_THERAPY,
            ('performance', 'compare_strategies'): JobType.BACKTEST_THERAPY,
            ('performance', 'profile_agents'): JobType.HOURLY_STATS,
            ('performance', 'resource_analysis'): JobType.HOURLY_STATS,
            ('performance', 'bottleneck_detection'): JobType.STATS_ANALYSIS,
            ('performance', 'database_health'): JobType.HOURLY_STATS,
            ('performance', 'disk_usage'): JobType.HOURLY_STATS,
            ('performance', 'memory_usage'): JobType.HOURLY_STATS,
            ('performance', 'optimization_task'): JobType.STATS_ANALYSIS,
            
            # Safety agent tasks
            ('safety', 'check_contraindications'): JobType.SAFETY_CHECK,
            ('safety', 'validate_dosing'): JobType.SAFETY_CHECK,
            ('safety', 'interaction_check'): JobType.SAFETY_CHECK,
            ('safety', 'validate_thresholds'): JobType.SAFETY_CHECK,
            ('safety', 'audit_decisions'): JobType.SAFETY_CHECK,
            ('safety', 'safety_task'): JobType.SAFETY_CHECK,
            
            # Nightly/reporting agent tasks
            ('nightly', 'generate_summary'): JobType.GENERATE_REPORT,
            ('nightly', 'aggregate_metrics'): JobType.HOURLY_STATS,
            ('nightly', 'daily_report'): JobType.GENERATE_REPORT,
            ('nightly', 'metrics_rollup'): JobType.HOURLY_STATS,
            ('nightly', 'lore_generation'): JobType.GENERATE_REPORT,
            ('nightly', 'backup_check'): JobType.HOURLY_STATS,
            ('nightly', 'reporting_task'): JobType.GENERATE_REPORT,
            
            # Coder agent tasks
            ('coder', 'code_todo'): JobType.HOURLY_STATS,
            ('coder', 'code_fixme'): JobType.HOURLY_STATS,
            ('coder', 'code_hack'): JobType.HOURLY_STATS,
            ('coder', 'code_xxx'): JobType.HOURLY_STATS,
        }
        
        # Try direct mapping first
        result = mapping.get((agent, task_type))
        if result:
            return result
        
        # Fallback logic based on agent type
        agent_defaults = {
            'rag': JobType.RAG_QUERY,
            'analytics': JobType.STATS_ANALYSIS,
            'ingestion': JobType.INGEST_TRIAL,
            'safety': JobType.SAFETY_CHECK,
            'performance': JobType.HOURLY_STATS,
            'nightly': JobType.GENERATE_REPORT,
            'enhancement': JobType.HOURLY_STATS,
            'coder': JobType.HOURLY_STATS,
        }
        
        return agent_defaults.get(agent, JobType.HOURLY_STATS)
    
    def check_logs_for_errors(self) -> List[str]:
        """Scan recent logs for errors"""
        errors = []
        
        if not self.log_dir.exists():
            return errors
        
        # Look for recent log files
        log_files = list(self.log_dir.glob("*.log"))
        
        for log_file in log_files[-5:]:  # Last 5 log files
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-50:]:  # Last 50 lines
                        if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'timeout']):
                            errors.append(f"{log_file.name}: {line.strip()}")
            except Exception:
                pass
        
        return errors[-10:]  # Return last 10 errors
    
    def detect_slow_agents(self, status: Dict[str, Any]) -> List[str]:
        """Detect agents that are taking too long"""
        slow = []
        
        for agent_info in status.get('agents', []):
            if agent_info.get('busy') and agent_info.get('current_job'):
                # Check job duration
                job_id = agent_info['current_job']
                job = self.governor.jobs.get(job_id)
                
                if job and job.started_at:
                    duration = (datetime.utcnow() - job.started_at).total_seconds()
                    
                    # Flag if running > 60 seconds
                    if duration > 60:
                        slow.append(f"{agent_info['name']} on {job_id} ({duration:.0f}s)")
                        self.slow_agents[agent_info['name']] = self.slow_agents.get(agent_info['name'], 0) + 1
        
        return slow
    
    def detect_anomalies(self, status: Dict[str, Any]) -> List[str]:
        """Detect system anomalies and degraded states"""
        anomalies = []
        
        # Check for queue backlog
        pending = status.get('pending', 0)
        if pending > 20:
            anomalies.append(f"⚠️  HIGH QUEUE BACKLOG: {pending} pending jobs")
        
        # Check for high failure rate
        completed = status.get('completed', 0)
        failed = status.get('failed', 0)
        if completed + failed > 0:
            failure_rate = failed / (completed + failed)
            if failure_rate > 0.3:
                anomalies.append(f"⚠️  HIGH FAILURE RATE: {failure_rate:.1%} ({failed}/{completed+failed})")
        
        # Check for stuck agents (busy for too long)
        for agent_info in status.get('agents', []):
            if agent_info.get('busy'):
                if agent_info['name'] in self.slow_agents and self.slow_agents[agent_info['name']] > 3:
                    anomalies.append(f"⚠️  DEGRADED AGENT: {agent_info['name']} (slow {self.slow_agents[agent_info['name']]} times)")
                    self.degraded_agents.add(agent_info['name'])
        
        return anomalies
    
    def propose_fixes(self, anomalies: List[str]) -> List[str]:
        """Propose fixes for detected issues"""
        proposals = []
        
        for anomaly in anomalies:
            if "HIGH QUEUE BACKLOG" in anomaly:
                proposals.append("💡 Suggestion: Consider spawning additional agent instances or increasing process_queue max_iterations")
            
            elif "HIGH FAILURE RATE" in anomaly:
                proposals.append("💡 Suggestion: Review recent errors in logs and consider agent prompt tuning")
            
            elif "DEGRADED AGENT" in anomaly:
                agent_name = anomaly.split(':')[1].split('(')[0].strip()
                proposals.append(f"💡 Suggestion: Restart or evolve agent: {agent_name}")
                self.evolution_suggestions.append(f"evolve_{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return proposals
    
    def log_cycle_metrics(self, status: Dict[str, Any], enqueued: int, errors: List[str]):
        """Log metrics for this cycle"""
        if not DUCKDB_AVAILABLE:
            return
        
        conn = duckdb.connect(self.db_path)
        
        # Calculate avg latency from recent jobs
        recent_jobs = [j for j in self.governor.jobs.values() 
                      if j.completed_at and j.started_at 
                      and (datetime.utcnow() - j.completed_at).total_seconds() < 300]
        
        avg_latency = 0.0
        if recent_jobs:
            latencies = [(j.completed_at - j.started_at).total_seconds() for j in recent_jobs]
            avg_latency = sum(latencies) / len(latencies)
        
        agents_busy = sum(1 for a in status.get('agents', []) if a.get('busy'))
        agents_idle = len(status.get('agents', [])) - agents_busy
        
        conn.execute("""
            INSERT INTO swarm_live_metrics VALUES (
                CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            self.cycle_count,
            status.get('pending', 0),
            status.get('running', 0),
            status.get('completed', 0),
            status.get('failed', 0),
            agents_busy,
            agents_idle,
            enqueued,
            avg_latency,
            len(errors),
            len(self.evolution_suggestions)
        ])
        
        conn.close()
    
    def generate_cycle_report(self, status: Dict[str, Any], enqueued: int, 
                             slow: List[str], anomalies: List[str], 
                             proposals: List[str], jobs_snapshot: List[Dict]) -> str:
        """Generate cycle report using configured formatter"""
        
        # Build cycle data for formatter
        cycle_data = {
            'cycle_number': self.cycle_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'enqueued': enqueued,
            'primary_agent': 'Oracle',  # Could be dynamic
            'jobs': jobs_snapshot,
            'agents': [
                {'name': a['name'], 'busy': a['busy']} 
                for a in status.get('agents', [])
            ],
        }
        
        # Use formatter
        formatted = self.formatter.format_cycle(cycle_data)
        
        # Add anomalies/proposals if present (append to formatted output)
        if slow or anomalies or proposals:
            extra_lines = []
            if slow:
                extra_lines.append("\n⏱️  Slow Agents:")
                extra_lines.extend(f"   {s}" for s in slow)
            if anomalies:
                extra_lines.append("\n⚠️  Anomalies:")
                extra_lines.extend(f"   {a}" for a in anomalies)
            if proposals:
                extra_lines.append("\n💡 Proposals:")
                extra_lines.extend(f"   {p}" for p in proposals)
            formatted += "\n" + "\n".join(extra_lines)
        
        return formatted
    
    def run_cycle(self):
        """Execute one monitoring and processing cycle"""
        self.cycle_count += 1
        
        # Capture job snapshot BEFORE processing (for report)
        jobs_snapshot = []
        
        try:
            # 1. Run task discovery every 60 seconds
            if (datetime.now() - self.last_discovery_cycle).total_seconds() > 60:
                discovery_result = self.task_discovery.discovery_cycle()
                if discovery_result['discovered'] > 0:
                    print(f"   ✅ Discovered {discovery_result['discovered']} new tasks")
                self.last_discovery_cycle = datetime.now()
            
            # 2. Auto-enqueue tasks from agent_tasks table (high volume for max utilization)
            enqueued = self.auto_enqueue_tasks(limit=50)  # Increased from 10
            
            # 3. Process job queue (max iterations for full CPU utilization)
            self.governor.process_queue(max_iterations=200)  # Increased from 50
            self.total_jobs_processed += enqueued
            
            # 4. Capture job details AFTER processing (for report)
            jobs_snapshot = [
                {
                    'job_id': job.job_id,
                    'stage': job.stage or 'unknown',
                    'status': job.status.value,
                    'duration': (job.completed_at - job.started_at).total_seconds() 
                                if job.completed_at and job.started_at else 0
                }
                for job in self.governor.jobs.values()
            ]
            
            # 5. Get status
            status = self.governor.status_report()
            
            # 6. Check logs for errors
            errors = self.check_logs_for_errors()
            if errors:
                self.last_errors = errors
            
            # 7. Detect slow agents
            slow = self.detect_slow_agents(status)
            
            # 8. Detect anomalies
            anomalies = self.detect_anomalies(status)
            
            # 9. Propose fixes
            proposals = self.propose_fixes(anomalies) if anomalies else []
            
            # 10. Log metrics
            self.log_cycle_metrics(status, enqueued, errors)
            
            # 11. Generate and print report
            report = self.generate_cycle_report(status, enqueued, slow, anomalies, proposals, jobs_snapshot)
            print("\n" + report)
            
            # 11. Write to log file
            log_file = self.log_dir / f"live_ops_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(report)
                f.write("\n")
        
        except Exception as e:
            print(f"❌ Cycle error: {e}")
            print(traceback.format_exc())
    
    def run_continuous(self):
        """Run continuous monitoring loop"""
        print("\n" + "="*80)
        print("🚀 SWARM LIVE OPS MODE - ACTIVATED")
        print("="*80)
        print(f"   Interval: {self.interval} seconds")
        print(f"   Database: {self.db_path}")
        print(f"   Logs: {self.log_dir}")
        print(f"   Output Style: {self.output_style}")
        print("\n   Running until CTRL+C...")
        print("="*80 + "\n")
        
        try:
            while True:
                self.run_cycle()
                
                print(f"💤 Sleeping {self.interval}s until next cycle...\n")
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("🛑 SWARM LIVE OPS - SHUTDOWN REQUESTED")
            print("="*80)
            print(f"   Total Cycles: {self.cycle_count}")
            print(f"   Total Jobs Processed: {self.total_jobs_processed}")
            print(f"   Uptime: {datetime.now() - self.start_time}")
            print(f"   Degraded Agents: {len(self.degraded_agents)}")
            if self.degraded_agents:
                print(f"      {', '.join(self.degraded_agents)}")
            print(f"   Evolution Suggestions: {len(self.evolution_suggestions)}")
            print("="*80 + "\n")
            
            # Final status
            self.governor.print_status()
            
            print("\n✅ Swarm Live Ops terminated gracefully.\n")


def main():
    parser = argparse.ArgumentParser(
        description='Swarm Live Ops - Continuous Autonomous Operation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python swarm_live_ops.py                    # Run with 10s interval
  python swarm_live_ops.py --interval 5       # Run with 5s interval
  python swarm_live_ops.py --once             # Single cycle only
  
Stop with CTRL+C
        """
    )
    
    parser.add_argument('--interval', type=int, default=10,
                       help='Seconds between cycles (default: 10)')
    parser.add_argument('--db', default='evidence.duckdb',
                       help='Database path (default: evidence.duckdb)')
    parser.add_argument('--style', default='compact',
                       choices=['compact', 'narrative', 'table', 'dashboard', 'json', 'minimal', 'dramatic'],
                       help='Output style (default: compact)')
    parser.add_argument('--once', action='store_true',
                       help='Run single cycle then exit')
    
    args = parser.parse_args()
    
    ops = SwarmLiveOps(interval=args.interval, db_path=args.db, output_style=args.style)
    
    if args.once:
        print("🔄 Running single cycle...\n")
        ops.run_cycle()
        print("\n✅ Single cycle complete.\n")
    else:
        ops.run_continuous()


if __name__ == '__main__':
    main()
