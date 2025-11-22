"""
Output formatters for swarm operations - turn assembly-line logs into different flavors.

Formatters:
- compact: Human-friendly summary
- narrative: Playful storytelling
- table: Clean tabular view
- dashboard: Status dashboard
- json: Machine-parsable
- minimal: Ultra-concise
- dramatic: Space-opera dispatch
"""
from __future__ import annotations

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta


class SwarmFormatter:
    """Base formatter with utility methods"""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    @staticmethod
    def job_summary(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract summary stats from job list"""
        completed = [j for j in jobs if j.get('status') == 'completed']
        failed = [j for j in jobs if j.get('status') == 'failed']
        running = [j for j in jobs if j.get('status') == 'running']
        pending = [j for j in jobs if j.get('status') == 'pending']
        
        durations = [j.get('duration', 0) for j in completed if j.get('duration')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total': len(jobs),
            'completed': len(completed),
            'failed': len(failed),
            'running': len(running),
            'pending': len(pending),
            'avg_duration': avg_duration,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
        }


class CompactFormatter(SwarmFormatter):
    """Human-friendly summary"""
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        
        lines = [
            f"\n🐝 Cycle #{cycle_data['cycle_number']} | {cycle_data['timestamp']}",
            "",
            "Queue Status:",
            f"  • {summary['completed']} completed, {summary['failed']} failed, {summary['pending']} pending",
            f"  • Average duration: {self.format_duration(summary['avg_duration'])}",
        ]
        
        if summary['completed'] > 0:
            lines.append(f"  • All tasks completed successfully" if summary['failed'] == 0 else f"  ⚠️  {summary['failed']} tasks failed")
        
        if cycle_data.get('enqueued', 0) > 0:
            lines.append(f"  • Auto-enqueued: {cycle_data['enqueued']} new tasks")
        
        return "\n".join(lines)


class NarrativeFormatter(SwarmFormatter):
    """Playful storytelling"""
    
    VERBS = ['munched', 'devoured', 'processed', 'analyzed', 'examined', 'digested']
    ADJECTIVES = ['diligently', 'swiftly', 'methodically', 'carefully', 'thoroughly']
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        agent_name = cycle_data.get('primary_agent', 'Oracle')
        
        import random
        verb = random.choice(self.VERBS)
        adj = random.choice(self.ADJECTIVES)
        
        lines = [
            f"\n📖 Cycle #{cycle_data['cycle_number']} - A Tale of Computation",
            "",
            f"The {agent_name} awoke, blinked at the queue, and {adj} began {verb.replace('ed', 'ing')} statistics.",
        ]
        
        if summary['completed'] > 0:
            lines.append(f"{summary['completed']} batches were {verb} through the pipeline.")
        
        if summary['failed'] > 0:
            lines.append(f"⚠️  {summary['failed']} brave tasks fell in battle, but the swarm persisted.")
        else:
            lines.append("Everything survived. No dragons were harmed.")
        
        if cycle_data.get('enqueued', 0) > 0:
            lines.append(f"Meanwhile, {cycle_data['enqueued']} fresh tasks arrived at the gates.")
        
        lines.append(f"\n⏱️  Average quest duration: {self.format_duration(summary['avg_duration'])}")
        
        return "\n".join(lines)


class TableFormatter(SwarmFormatter):
    """Clean tabular view"""
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        jobs = cycle_data.get('jobs', [])[:20]  # Limit to first 20
        
        lines = [
            f"\n📊 Cycle #{cycle_data['cycle_number']} | {cycle_data['timestamp']}",
            "",
            f"{'Job ID':<20} | {'Stage':<12} | {'Status':<10} | {'Duration':<10}",
            "-" * 65,
        ]
        
        for job in jobs:
            job_id = job.get('job_id', 'unknown')[-12:]  # Last 12 chars
            stage = job.get('stage', 'unknown')[:12]
            status = job.get('status', 'unknown')[:10]
            duration = self.format_duration(job.get('duration', 0)) if job.get('duration') else '-'
            
            lines.append(f"...{job_id:<17} | {stage:<12} | {status:<10} | {duration:<10}")
        
        if len(cycle_data.get('jobs', [])) > 20:
            lines.append(f"... ({len(cycle_data['jobs']) - 20} more jobs)")
        
        summary = self.job_summary(cycle_data.get('jobs', []))
        lines.extend([
            "",
            f"Summary: {summary['completed']} completed, {summary['failed']} failed, {summary['pending']} pending",
        ])
        
        return "\n".join(lines)


class DashboardFormatter(SwarmFormatter):
    """Status dashboard"""
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        
        health = "GOOD" if summary['failed'] == 0 else "DEGRADED" if summary['failed'] < 3 else "CRITICAL"
        health_icon = "✅" if health == "GOOD" else "⚠️" if health == "DEGRADED" else "🔴"
        
        lines = [
            f"\n🎛️  SWARM DASHBOARD | Cycle #{cycle_data['cycle_number']}",
            "=" * 60,
            "",
            f"[{health_icon}] System Health: {health}",
            "",
            "[Status]",
            f"  Completed: {summary['completed']}",
            f"  Failed: {summary['failed']}",
            f"  Running: {summary['running']}",
            f"  Pending: {summary['pending']}",
            "",
            "[Metrics]",
            f"  Duration (avg): {self.format_duration(summary['avg_duration'])}",
            f"  Duration (min): {self.format_duration(summary['min_duration'])}",
            f"  Duration (max): {self.format_duration(summary['max_duration'])}",
        ]
        
        if cycle_data.get('enqueued', 0) > 0:
            lines.extend([
                "",
                "[Activity]",
                f"  New Tasks: {cycle_data['enqueued']}",
            ])
        
        agents = cycle_data.get('agents', [])
        busy = sum(1 for a in agents if a.get('busy'))
        if agents:
            lines.extend([
                "",
                "[Agents]",
                f"  Active: {busy}/{len(agents)}",
            ])
        
        return "\n".join(lines)


class JSONFormatter(SwarmFormatter):
    """Machine-parsable JSON"""
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        
        output = {
            "cycle": cycle_data['cycle_number'],
            "timestamp": cycle_data['timestamp'],
            "summary": summary,
            "enqueued": cycle_data.get('enqueued', 0),
            "agents": cycle_data.get('agents', []),
        }
        
        return json.dumps(output, indent=2)


class MinimalFormatter(SwarmFormatter):
    """Ultra-concise, one-liner"""
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        status = "✅" if summary['failed'] == 0 else "⚠️"
        
        return (
            f"{status} C{cycle_data['cycle_number']}: "
            f"{summary['completed']}✅ {summary['failed']}❌ "
            f"~{self.format_duration(summary['avg_duration'])} "
            f"[+{cycle_data.get('enqueued', 0)} enqueued]"
        )


class DramaticFormatter(SwarmFormatter):
    """Space-opera dispatch"""
    
    OPENINGS = [
        "STARDATE {stardate} - Sector Command reports:",
        "TRANSMISSION RECEIVED - Cycle {cycle}:",
        "⚡ FLEET STATUS UPDATE - Mission {cycle}:",
        "🚀 OPERATIONAL BRIEF - Campaign {cycle}:",
    ]
    
    def format_cycle(self, cycle_data: Dict[str, Any]) -> str:
        summary = self.job_summary(cycle_data.get('jobs', []))
        
        import random
        opening = random.choice(self.OPENINGS).format(
            stardate=cycle_data['cycle_number'] + 40000,
            cycle=cycle_data['cycle_number']
        )
        
        lines = [
            f"\n{opening}",
            "",
        ]
        
        if summary['completed'] > 0:
            lines.append(f"⚔️  {summary['completed']} objectives secured. Mission parameters: NOMINAL.")
        
        if summary['failed'] > 0:
            lines.append(f"🔴 ALERT: {summary['failed']} units reported losses. Reinforcements en route.")
        else:
            lines.append("🛡️  All units returned safely. Zero casualties.")
        
        if cycle_data.get('enqueued', 0) > 0:
            lines.append(f"📡 {cycle_data['enqueued']} new directives received from Central Command.")
        
        lines.extend([
            "",
            f"⏱️  Average engagement time: {self.format_duration(summary['avg_duration'])}",
            f"🎖️  Fleet efficiency: {'EXEMPLARY' if summary['failed'] == 0 else 'ACCEPTABLE'}",
            "",
            "--- END TRANSMISSION ---",
        ])
        
        return "\n".join(lines)


# Factory function
def get_formatter(style: str = 'compact') -> SwarmFormatter:
    """Get formatter by style name"""
    formatters = {
        'compact': CompactFormatter(),
        'narrative': NarrativeFormatter(),
        'table': TableFormatter(),
        'dashboard': DashboardFormatter(),
        'json': JSONFormatter(),
        'minimal': MinimalFormatter(),
        'dramatic': DramaticFormatter(),
    }
    
    return formatters.get(style.lower(), CompactFormatter())


# CLI demo
if __name__ == '__main__':
    # Sample cycle data
    sample_data = {
        'cycle_number': 42,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'enqueued': 12,
        'primary_agent': 'Oracle',
        'jobs': [
            {'job_id': 'job_123_abc', 'stage': 'finalize', 'status': 'completed', 'duration': 31.2},
            {'job_id': 'job_456_def', 'stage': 'finalize', 'status': 'completed', 'duration': 28.5},
            {'job_id': 'job_789_ghi', 'stage': 'aggregate', 'status': 'completed', 'duration': 15.3},
            {'job_id': 'job_012_jkl', 'stage': 'init', 'status': 'pending', 'duration': 0},
        ],
        'agents': [
            {'name': 'Oracle', 'busy': False},
            {'name': 'Harvester', 'busy': True},
        ],
    }
    
    styles = ['compact', 'narrative', 'table', 'dashboard', 'json', 'minimal', 'dramatic']
    
    for style in styles:
        formatter = get_formatter(style)
        print(f"\n{'='*70}")
        print(f"STYLE: {style.upper()}")
        print('='*70)
        print(formatter.format_cycle(sample_data))
