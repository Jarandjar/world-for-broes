#!/usr/bin/env python3
"""
Daily Briefing Engine - Your Morning Command Scroll

Combines:
- Task status (completed yesterday, pending today)
- Fitness intel (training load, recovery status)
- System health (agent performance, anomalies)
- Mythic narrative wrapping

Output: Single killer report to start your day.
"""

import duckdb
from datetime import datetime, timedelta
from pathlib import Path
import json
from agent_sdk import log_event, log_report


class DailyBriefingEngine:
    """Generates comprehensive morning briefings"""
    
    def __init__(self, db_path: str = 'evidence.duckdb'):
        self.db_path = db_path
        self.output_dir = Path('out/briefings')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_briefing(self) -> dict:
        """Generate complete daily briefing"""
        conn = duckdb.connect(self.db_path)
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        briefing = {
            'timestamp': now.isoformat(),
            'date_human': now.strftime('%A, %B %d, %Y'),
            'tasks': self._get_task_status(conn, yesterday),
            'fitness': self._get_fitness_status(conn),
            'system': self._get_system_health(conn, yesterday),
            'priorities': self._generate_priorities(conn),
            'risks': self._identify_risks(conn)
        }
        
        conn.close()
        
        # Generate narrative
        briefing['narrative'] = self._weave_narrative(briefing)
        
        # Save report
        self._save_briefing(briefing)
        
        # Log to event spine
        log_report(
            agent_id='daily_briefing',
            realm='command_center',
            report_text=briefing['narrative'],
            metadata={
                'tasks_completed': briefing['tasks']['completed_yesterday'],
                'tasks_pending': briefing['tasks']['pending_today'],
                'fitness_status': briefing['fitness']['status'],
                'system_health': briefing['system']['health_score']
            }
        )
        
        return briefing
    
    def _get_task_status(self, conn, yesterday) -> dict:
        """Analyze task completion and pending work"""
        
        # Completed yesterday
        completed = conn.execute("""
            SELECT COUNT(*) 
            FROM agent_tasks 
            WHERE status = 'completed' 
              AND completed_at >= ?
        """, [yesterday]).fetchone()[0]
        
        # Pending high priority
        pending_high = conn.execute("""
            SELECT COUNT(*) 
            FROM agent_tasks 
            WHERE status = 'pending' 
              AND priority >= 7
        """).fetchone()[0]
        
        # Total pending
        pending_total = conn.execute("""
            SELECT COUNT(*) 
            FROM agent_tasks 
            WHERE status = 'pending'
        """).fetchone()[0]
        
        # Top task types pending
        top_types = conn.execute("""
            SELECT task_type, COUNT(*) as cnt
            FROM agent_tasks
            WHERE status = 'pending'
            GROUP BY task_type
            ORDER BY cnt DESC
            LIMIT 5
        """).fetchall()
        
        return {
            'completed_yesterday': completed,
            'pending_today': pending_total,
            'pending_high_priority': pending_high,
            'top_types': [(t[0], t[1]) for t in top_types]
        }
    
    def _get_fitness_status(self, conn) -> dict:
        """Get fitness training status"""
        try:
            # Check if we have recent fitness data
            recent = conn.execute("""
                SELECT COUNT(*) 
                FROM workouts 
                WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """).fetchone()[0]
            
            if recent == 0:
                return {
                    'status': 'no_data',
                    'message': 'No recent workout data'
                }
            
            # Get weekly volume
            weekly_hours = conn.execute("""
                SELECT SUM(duration_seconds)/3600.0 as hours
                FROM workouts
                WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '7 days'
            """).fetchone()[0] or 0
            
            # Estimate training load (simplified)
            status = 'recovering' if weekly_hours < 3 else 'training' if weekly_hours < 10 else 'heavy_load'
            
            return {
                'status': status,
                'weekly_hours': round(weekly_hours, 1),
                'workouts_this_week': recent,
                'message': f'{weekly_hours:.1f}h training this week'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_system_health(self, conn, yesterday) -> dict:
        """Analyze swarm system health"""
        
        # Tasks processed in last 24h
        processed_24h = conn.execute("""
            SELECT COUNT(*)
            FROM agent_tasks
            WHERE started_at >= ?
        """, [yesterday]).fetchone()[0]
        
        # Error rate
        failed_24h = conn.execute("""
            SELECT COUNT(*)
            FROM agent_tasks
            WHERE status = 'failed'
              AND started_at >= ?
        """, [yesterday]).fetchone()[0]
        
        error_rate = (failed_24h / processed_24h * 100) if processed_24h > 0 else 0
        
        # Agent activity
        active_agents = conn.execute("""
            SELECT COUNT(DISTINCT agent_name)
            FROM agent_tasks
            WHERE started_at >= ?
        """, [yesterday]).fetchone()[0]
        
        # Health score (0-100)
        health_score = 100
        if error_rate > 10:
            health_score -= 30
        elif error_rate > 5:
            health_score -= 15
        
        if processed_24h < 10:
            health_score -= 20  # Low activity
        
        return {
            'health_score': health_score,
            'processed_24h': processed_24h,
            'error_rate': round(error_rate, 1),
            'active_agents': active_agents,
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'critical'
        }
    
    def _generate_priorities(self, conn) -> list:
        """Generate today's priority list"""
        priorities = conn.execute("""
            SELECT task_type, agent_name, priority, task_data
            FROM agent_tasks
            WHERE status = 'pending'
              AND priority >= 7
            ORDER BY priority DESC, created_at ASC
            LIMIT 10
        """).fetchall()
        
        return [
            {
                'task_type': p[0],
                'agent': p[1],
                'priority': p[2],
                'description': json.loads(p[3]).get('description', p[0]) if p[3] else p[0]
            }
            for p in priorities
        ]
    
    def _identify_risks(self, conn) -> list:
        """Identify potential risks/blockers"""
        risks = []
        
        # Check for task backlog
        pending_count = conn.execute("""
            SELECT COUNT(*) FROM agent_tasks WHERE status = 'pending'
        """).fetchone()[0]
        
        if pending_count > 500:
            risks.append({
                'type': 'task_overload',
                'severity': 'high',
                'message': f'Task queue overloaded: {pending_count} pending tasks'
            })
        elif pending_count > 200:
            risks.append({
                'type': 'task_buildup',
                'severity': 'medium',
                'message': f'Task queue building up: {pending_count} pending tasks'
            })
        
        # Check for stuck tasks (pending > 24h)
        stuck = conn.execute("""
            SELECT COUNT(*)
            FROM agent_tasks
            WHERE status = 'pending'
              AND created_at < CURRENT_TIMESTAMP - INTERVAL '24 hours'
        """).fetchone()[0]
        
        if stuck > 50:
            risks.append({
                'type': 'stuck_tasks',
                'severity': 'medium',
                'message': f'{stuck} tasks stuck for >24h'
            })
        
        return risks
    
    def _weave_narrative(self, briefing: dict) -> str:
        """Generate mythic narrative for briefing"""
        
        date = briefing['date_human']
        tasks = briefing['tasks']
        fitness = briefing['fitness']
        system = briefing['system']
        priorities = briefing['priorities']
        risks = briefing['risks']
        
        # Status emoji
        status_emoji = {
            'healthy': '✨',
            'degraded': '⚠️',
            'critical': '🔥'
        }.get(system['status'], '⚙️')
        
        narrative = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║                   📜 DAILY COMMAND SCROLL 📜                      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

**{date}**

## 🌅 Yesterday's Victories

The swarm processed **{system['processed_24h']} tasks** across **{system['active_agents']} active agents**.
You conquered **{tasks['completed_yesterday']} missions** in the last cycle.

**Error Rate:** {system['error_rate']}% {'(All systems nominal)' if system['error_rate'] < 5 else '(Minor turbulence detected)'}

## 🎯 Today's Campaign

**Active Quests:** {tasks['pending_today']} tasks in queue
**High Priority Missions:** {tasks['pending_high_priority']} urgent objectives

"""
        
        # Top priorities
        if priorities:
            narrative += "### ⚔️ Priority Missions\n\n"
            for i, p in enumerate(priorities[:5], 1):
                priority_icon = '🔴' if p['priority'] >= 9 else '🟡' if p['priority'] >= 7 else '🟢'
                narrative += f"{i}. {priority_icon} **{p['task_type']}** (P{p['priority']}) - {p['agent']}\n"
            narrative += "\n"
        
        # Fitness status
        if fitness['status'] != 'no_data':
            fitness_icon = '💪' if fitness['status'] == 'training' else '🧘' if fitness['status'] == 'recovering' else '🔥'
            narrative += f"## {fitness_icon} Physical Realm Status\n\n"
            narrative += f"**Training Load:** {fitness['message']}\n"
            narrative += f"**Workouts Completed:** {fitness.get('workouts_this_week', 0)} this week\n\n"
            
            if fitness['status'] == 'heavy_load':
                narrative += "⚠️  *Heavy training detected. The Body Guardian suggests rest rituals.*\n\n"
            elif fitness['status'] == 'recovering':
                narrative += "✨ *Recovery phase active. Light missions recommended.*\n\n"
        
        # Risks/warnings
        if risks:
            narrative += "## ⚠️  Detected Threats\n\n"
            for risk in risks:
                severity_icon = '🔴' if risk['severity'] == 'high' else '🟡'
                narrative += f"- {severity_icon} **{risk['type'].upper()}**: {risk['message']}\n"
            narrative += "\n"
        
        # System health
        narrative += f"## {status_emoji} Swarm Consciousness\n\n"
        narrative += f"**Health Score:** {system['health_score']}/100\n"
        narrative += f"**Status:** {system['status'].upper()}\n\n"
        
        # Closing
        narrative += """
═══════════════════════════════════════════════════════════════════

**The swarm awaits your command.**

*Run: `python swarm_live_ops.py --style narrative` to begin operations.*

═══════════════════════════════════════════════════════════════════
"""
        
        return narrative
    
    def _save_briefing(self, briefing: dict):
        """Save briefing to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save narrative
        narrative_file = self.output_dir / f'briefing_{timestamp}.md'
        narrative_file.write_text(briefing['narrative'])
        
        # Save JSON for programmatic access
        json_file = self.output_dir / f'briefing_{timestamp}.json'
        json_file.write_text(json.dumps(briefing, indent=2))
        
        print(f"✅ Briefing saved:")
        print(f"   📜 {narrative_file}")
        print(f"   📊 {json_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Briefing Engine')
    parser.add_argument('--db', default='evidence.duckdb', help='Database path')
    parser.add_argument('--print', action='store_true', help='Print briefing to console')
    
    args = parser.parse_args()
    
    print("📜 Generating daily briefing...\n")
    
    engine = DailyBriefingEngine(db_path=args.db)
    briefing = engine.generate_briefing()
    
    if args.print:
        print(briefing['narrative'])
    else:
        print("\n✨ Briefing complete!")
        print(f"   Tasks: {briefing['tasks']['pending_today']} pending, {briefing['tasks']['completed_yesterday']} done yesterday")
        print(f"   System: {briefing['system']['status']} ({briefing['system']['health_score']}/100)")
        print(f"   Fitness: {briefing['fitness']['status']}")


if __name__ == '__main__':
    main()
