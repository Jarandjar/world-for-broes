#!/usr/bin/env python3
"""
Agent: Task Discovery & Auto-Harvesting

Continuously scans the project for new work:
- Monitors MASTER_TODO.md for changes
- Scans code for new TODOs/FIXMEs
- Watches for new files that need processing
- Generates maintenance tasks based on system state
- Auto-populates agent_tasks table

This agent keeps the swarm fed with work autonomously.
"""
from __future__ import annotations

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set
import json

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from swarm_task_harvester import TaskHarvester


class TaskDiscoveryAgent:
    """Autonomous task discovery and harvesting agent"""
    
    def __init__(self, db_path: str = 'evidence.duckdb', project_root: Path = None):
        self.db_path = db_path
        self.project_root = project_root or Path.cwd()
        self.harvester = TaskHarvester(db_path, project_root)
        
        # State tracking
        self.last_todo_hash = None
        self.last_code_scan = datetime.now()
        self.last_maintenance_gen = datetime.now()
        self.known_files: Set[str] = set()
        
        # Statistics
        self.total_discovered = 0
        self.cycles_run = 0
        
        print("\n🔍 Task Discovery Agent Initialized")
        print(f"   Project: {self.project_root}")
        print(f"   Database: {self.db_path}")
    
    def get_file_hash(self, filepath: Path) -> str:
        """Get MD5 hash of file content"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def check_todo_changes(self) -> bool:
        """Check if MASTER_TODO.md has changed"""
        todo_file = self.project_root / "MASTER_TODO.md"
        
        if not todo_file.exists():
            return False
        
        current_hash = self.get_file_hash(todo_file)
        
        if self.last_todo_hash is None:
            self.last_todo_hash = current_hash
            return True  # First run
        
        if current_hash != self.last_todo_hash:
            print(f"\n📝 MASTER_TODO.md changed (old: {self.last_todo_hash[:8]}, new: {current_hash[:8]})")
            self.last_todo_hash = current_hash
            return True
        
        return False
    
    def scan_for_new_files(self) -> List[Dict]:
        """Scan for new Python files that need analysis"""
        new_files = []
        
        py_files = list(self.project_root.glob("*.py"))
        
        for py_file in py_files:
            file_str = str(py_file)
            if file_str not in self.known_files:
                self.known_files.add(file_str)
                
                # Generate analysis task for new file
                new_files.append({
                    'agent_name': 'coder',
                    'task_type': 'analyze_new_file',
                    'task_data': {
                        'file': py_file.name,
                        'discovered_at': datetime.now().isoformat(),
                        'source': 'file_discovery'
                    },
                    'priority': 5
                })
        
        if new_files:
            print(f"\n🆕 Discovered {len(new_files)} new files")
        
        return new_files
    
    def check_queue_depth(self) -> int:
        """Check how many pending tasks remain"""
        if not DUCKDB_AVAILABLE:
            return 0
        
        conn = duckdb.connect(self.db_path)
        
        try:
            count = conn.execute("""
                SELECT COUNT(*) 
                FROM agent_tasks 
                WHERE status = 'pending'
            """).fetchone()[0]
        except Exception:
            count = 0
        
        conn.close()
        return count
    
    def generate_periodic_tasks(self) -> List[Dict]:
        """Generate recurring maintenance and health check tasks"""
        tasks = []
        now = datetime.now()
        
        # Check if it's time for maintenance tasks (every 5 minutes)
        if (now - self.last_maintenance_gen).total_seconds() > 300:
            print("\n🔧 Generating periodic maintenance tasks...")
            
            tasks.extend([
                {
                    'agent_name': 'performance',
                    'task_type': 'health_check',
                    'task_data': {
                        'check_type': 'system_health',
                        'generated_at': now.isoformat(),
                        'source': 'periodic_maintenance'
                    },
                    'priority': 7
                },
                {
                    'agent_name': 'rag',
                    'task_type': 'cache_stats',
                    'task_data': {
                        'check_type': 'cache_health',
                        'generated_at': now.isoformat(),
                        'source': 'periodic_maintenance'
                    },
                    'priority': 6
                },
                {
                    'agent_name': 'nightly',
                    'task_type': 'metrics_snapshot',
                    'task_data': {
                        'snapshot_type': 'periodic',
                        'generated_at': now.isoformat(),
                        'source': 'periodic_maintenance'
                    },
                    'priority': 5
                },
            ])
            
            # Weekly fitness sync (Mondays at 6am)
            if now.weekday() == 0 and now.hour == 6:
                tasks.append({
                    'agent_name': 'fitness_harvester',
                    'task_type': 'fitness_sync',
                    'task_data': {
                        'days_back': 7,
                        'generated_at': now.isoformat(),
                        'source': 'weekly_sync'
                    },
                    'priority': 6
                })
                print("   📅 Weekly fitness sync scheduled")
            
            # Daily fitness analysis (every day at 7am)
            if now.hour == 7:
                tasks.append({
                    'agent_name': 'fitness_analyst',
                    'task_type': 'fitness_analysis',
                    'task_data': {
                        'analysis_type': 'full_report',
                        'generated_at': now.isoformat(),
                        'source': 'daily_analysis'
                    },
                    'priority': 5
                })
                print("   📊 Daily fitness analysis scheduled")
            
            self.last_maintenance_gen = now
            print(f"   ✅ Generated {len(tasks)} maintenance tasks")
        
        return tasks
    
    def generate_adaptive_tasks(self) -> List[Dict]:
        """Generate tasks based on system state and queue depth"""
        tasks = []
        queue_depth = self.check_queue_depth()
        
        # If queue is getting low, generate more work
        if queue_depth < 20:
            print(f"\n⚠️  Queue low ({queue_depth} tasks) - Generating adaptive work...")
            
            # Generate exploration tasks
            tasks.extend([
                {
                    'agent_name': 'analytics',
                    'task_type': 'exploratory_analysis',
                    'task_data': {
                        'analysis_type': 'data_quality',
                        'reason': 'queue_replenishment',
                        'generated_at': datetime.now().isoformat()
                    },
                    'priority': 6
                },
                {
                    'agent_name': 'rag',
                    'task_type': 'index_optimization',
                    'task_data': {
                        'optimization_type': 'reindex',
                        'reason': 'queue_replenishment',
                        'generated_at': datetime.now().isoformat()
                    },
                    'priority': 6
                },
                {
                    'agent_name': 'enhancement',
                    'task_type': 'scan_improvements',
                    'task_data': {
                        'scan_type': 'code_quality',
                        'reason': 'queue_replenishment',
                        'generated_at': datetime.now().isoformat()
                    },
                    'priority': 5
                },
            ])
            
            print(f"   ✅ Generated {len(tasks)} adaptive tasks")
        
        return tasks
    
    def insert_discovered_tasks(self, tasks: List[Dict]) -> int:
        """Insert discovered tasks into agent_tasks table"""
        if not tasks or not DUCKDB_AVAILABLE:
            return 0
        
        conn = duckdb.connect(self.db_path)
        
        # Get max task_id
        try:
            max_id = conn.execute("SELECT MAX(task_id) FROM agent_tasks").fetchone()[0] or 0
        except:
            max_id = 0
        
        inserted = 0
        for i, task in enumerate(tasks, start=max_id + 1):
            try:
                conn.execute("""
                    INSERT INTO agent_tasks (
                        task_id, agent_name, task_type, task_data, 
                        status, priority, created_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, CURRENT_TIMESTAMP)
                """, [
                    i,
                    task['agent_name'],
                    task['task_type'],
                    json.dumps(task['task_data']),
                    task['priority']
                ])
                inserted += 1
            except Exception as e:
                print(f"   ⚠️  Failed to insert task: {e}")
        
        conn.close()
        return inserted
    
    def discovery_cycle(self) -> Dict:
        """Run one discovery cycle"""
        self.cycles_run += 1
        print(f"\n{'='*80}")
        print(f"🔍 TASK DISCOVERY CYCLE #{self.cycles_run}")
        print(f"{'='*80}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        discovered_tasks = []
        
        # 1. Check for TODO changes
        if self.check_todo_changes():
            print("\n📋 Harvesting MASTER_TODO.md...")
            todo_tasks = self.harvester.harvest_master_todo()
            discovered_tasks.extend(todo_tasks)
        
        # 2. Scan for new code TODOs (every 2 minutes)
        if (datetime.now() - self.last_code_scan).total_seconds() > 120:
            print("\n🔍 Scanning code for new TODOs...")
            code_tasks = self.harvester.harvest_code_todos()
            discovered_tasks.extend(code_tasks)
            self.last_code_scan = datetime.now()
        
        # 3. Scan for new files
        new_file_tasks = self.scan_for_new_files()
        discovered_tasks.extend(new_file_tasks)
        
        # 4. Generate periodic maintenance tasks
        maintenance_tasks = self.generate_periodic_tasks()
        discovered_tasks.extend(maintenance_tasks)
        
        # 5. Generate adaptive tasks based on queue depth
        adaptive_tasks = self.generate_adaptive_tasks()
        discovered_tasks.extend(adaptive_tasks)
        
        # Insert all discovered tasks
        if discovered_tasks:
            inserted = self.insert_discovered_tasks(discovered_tasks)
            self.total_discovered += inserted
            
            print(f"\n📊 Discovery Summary:")
            print(f"   Tasks Discovered: {len(discovered_tasks)}")
            print(f"   Tasks Inserted: {inserted}")
            print(f"   Total Lifetime: {self.total_discovered}")
        else:
            print("\n💤 No new tasks discovered this cycle")
        
        # Queue status
        queue_depth = self.check_queue_depth()
        print(f"\n📈 Current Queue: {queue_depth} pending tasks")
        
        print(f"{'='*80}\n")
        
        return {
            'cycle': self.cycles_run,
            'discovered': len(discovered_tasks),
            'inserted': inserted if discovered_tasks else 0,
            'queue_depth': queue_depth,
            'total_lifetime': self.total_discovered
        }
    
    def run_continuous(self, interval: int = 60):
        """Run continuous discovery loop"""
        print("\n" + "="*80)
        print("🔍 TASK DISCOVERY AGENT - CONTINUOUS MODE")
        print("="*80)
        print(f"   Scan Interval: {interval} seconds")
        print(f"   TODO Monitor: Active")
        print(f"   Code Scanner: Every 2 minutes")
        print(f"   File Watcher: Active")
        print(f"   Adaptive Generation: Queue < 20")
        print("\n   Running until CTRL+C...")
        print("="*80 + "\n")
        
        try:
            while True:
                result = self.discovery_cycle()
                
                print(f"💤 Sleeping {interval}s until next scan...\n")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("🛑 TASK DISCOVERY AGENT - SHUTDOWN")
            print("="*80)
            print(f"   Total Cycles: {self.cycles_run}")
            print(f"   Total Tasks Discovered: {self.total_discovered}")
            print("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Task Discovery Agent - Autonomous work finder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_task_discovery.py                    # Run with 60s interval
  python agent_task_discovery.py --interval 30      # Run with 30s interval
  python agent_task_discovery.py --once             # Single discovery cycle
  
Stop with CTRL+C
        """
    )
    
    parser.add_argument('--interval', type=int, default=60,
                       help='Seconds between discovery cycles (default: 60)')
    parser.add_argument('--db', default='evidence.duckdb',
                       help='Database path (default: evidence.duckdb)')
    parser.add_argument('--once', action='store_true',
                       help='Run single discovery cycle then exit')
    parser.add_argument('--project-root', type=Path, default=None,
                       help='Project root directory')
    
    args = parser.parse_args()
    
    agent = TaskDiscoveryAgent(
        db_path=args.db,
        project_root=args.project_root
    )
    
    if args.once:
        print("🔍 Running single discovery cycle...\n")
        result = agent.discovery_cycle()
        print(f"\n✅ Discovery complete: {result['discovered']} tasks found, {result['inserted']} inserted\n")
    else:
        agent.run_continuous(interval=args.interval)


if __name__ == '__main__':
    main()
