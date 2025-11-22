#!/usr/bin/env python3
"""
Fitness Analyst Agent
Analyzes workout data for training insights, overtraining detection, performance trends
"""

import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from agent_sdk import log_event, log_report

@dataclass
class TrainingMetrics:
    """Weekly training metrics"""
    week_start: datetime
    total_workouts: int
    total_hours: float
    total_km: float
    avg_heart_rate: Optional[int]
    workout_types: List[str]
    chronic_load: float  # 42-day rolling average
    acute_load: float    # 7-day average
    training_stress_balance: float  # CTL - ATL


class FitnessAnalyst:
    """Agent for workout data analysis"""
    
    def __init__(self, db_path: str = 'evidence.duckdb'):
        self.db_path = db_path
        print("📊 Fitness Analyst Agent Initialized")
    
    def calculate_training_load(self, days_back: int = 90) -> List[Dict[str, Any]]:
        """Calculate chronic (CTL) and acute (ATL) training load"""
        conn = duckdb.connect(self.db_path)
        
        # Training load = duration * intensity (HR-based)
        # CTL = 42-day exponential weighted moving average
        # ATL = 7-day exponential weighted moving average
        
        query = """
        WITH daily_load AS (
            SELECT 
                start_time::DATE as day,
                SUM(duration_seconds * COALESCE(avg_heart_rate, 140) / 3600.0) as daily_load
            FROM workouts
            WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '90 days'
            GROUP BY day
        ),
        ctl_calc AS (
            SELECT 
                day,
                daily_load,
                AVG(daily_load) OVER (
                    ORDER BY day 
                    ROWS BETWEEN 41 PRECEDING AND CURRENT ROW
                ) as ctl,
                AVG(daily_load) OVER (
                    ORDER BY day 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as atl
            FROM daily_load
        )
        SELECT 
            day,
            daily_load,
            ctl,
            atl,
            ctl - atl as tsb
        FROM ctl_calc
        ORDER BY day DESC
        LIMIT ?
        """
        
        results = conn.execute(query, [days_back]).fetchall()
        conn.close()
        
        metrics = []
        for row in results:
            metrics.append({
                'day': row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                'daily_load': round(row[1], 1) if row[1] else 0,
                'ctl': round(row[2], 1) if row[2] else 0,
                'atl': round(row[3], 1) if row[3] else 0,
                'tsb': round(row[4], 1) if row[4] else 0
            })
        
        return metrics
    
    def detect_overtraining(self) -> Dict[str, Any]:
        """Detect signs of overtraining or fatigue"""
        conn = duckdb.connect(self.db_path)
        
        # Indicators:
        # 1. High volume + declining average HR (fatigue)
        # 2. Negative TSB for >2 weeks (accumulated fatigue)
        # 3. Declining performance (slower paces at same HR)
        
        # Check recent 2 weeks vs previous 2 weeks
        query = """
        WITH recent AS (
            SELECT 
                'recent' as period,
                COUNT(*) as workouts,
                SUM(duration_seconds)/3600 as hours,
                AVG(avg_heart_rate) as avg_hr,
                AVG(avg_pace_min_per_km) as avg_pace
            FROM workouts
            WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '14 days'
              AND activity_type IN ('run', 'ride')
              AND avg_heart_rate IS NOT NULL
        ),
        previous AS (
            SELECT 
                'previous' as period,
                COUNT(*) as workouts,
                SUM(duration_seconds)/3600 as hours,
                AVG(avg_heart_rate) as avg_hr,
                AVG(avg_pace_min_per_km) as avg_pace
            FROM workouts
            WHERE start_time BETWEEN 
                  CURRENT_TIMESTAMP - INTERVAL '28 days' AND 
                  CURRENT_TIMESTAMP - INTERVAL '14 days'
              AND activity_type IN ('run', 'ride')
              AND avg_heart_rate IS NOT NULL
        )
        SELECT * FROM recent
        UNION ALL
        SELECT * FROM previous
        """
        
        results = conn.execute(query).fetchall()
        conn.close()
        
        if len(results) < 2:
            return {'overtraining_risk': 'insufficient_data'}
        
        recent = results[0]
        previous = results[1]
        
        # Calculate changes
        volume_change = (recent[2] - previous[2]) / previous[2] if previous[2] else 0
        hr_change = (recent[3] - previous[3]) / previous[3] if previous[3] else 0
        pace_change = (recent[4] - previous[4]) / previous[4] if recent[4] and previous[4] else 0
        
        risk_factors = []
        risk_level = 'low'
        
        # High volume + low HR = fatigue
        if volume_change > 0.3 and hr_change < -0.05:
            risk_factors.append('high_volume_low_hr')
            risk_level = 'moderate'
        
        # Declining HR at same volume
        if abs(volume_change) < 0.1 and hr_change < -0.1:
            risk_factors.append('declining_hr')
            risk_level = 'moderate'
        
        # Slower pace at same/higher HR (performance decline)
        if pace_change > 0.1 and hr_change >= 0:
            risk_factors.append('performance_decline')
            risk_level = 'high' if 'declining_hr' in risk_factors else 'moderate'
        
        return {
            'overtraining_risk': risk_level,
            'risk_factors': risk_factors,
            'recent_hours': round(recent[2], 1) if recent[2] else 0,
            'previous_hours': round(previous[2], 1) if previous[2] else 0,
            'volume_change_pct': round(volume_change * 100, 1),
            'hr_change_pct': round(hr_change * 100, 1),
            'pace_change_pct': round(pace_change * 100, 1),
            'recommendation': self._get_overtraining_recommendation(risk_level, risk_factors)
        }
    
    def _get_overtraining_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Generate training recommendation based on risk"""
        if risk_level == 'high':
            return "⚠️ High overtraining risk detected. Consider 2-3 rest days and reduced volume."
        elif risk_level == 'moderate':
            if 'high_volume_low_hr' in factors:
                return "Consider a recovery week with 30% reduced volume."
            elif 'performance_decline' in factors:
                return "Performance declining. Add more easy days and prioritize sleep."
            else:
                return "Monitor closely. Ensure adequate recovery between hard sessions."
        else:
            return "✅ Training load appears manageable."
    
    def analyze_weekly_patterns(self, weeks: int = 12) -> List[TrainingMetrics]:
        """Analyze weekly training patterns"""
        conn = duckdb.connect(self.db_path)
        
        cutoff = datetime.now() - timedelta(weeks=weeks)
        
        query = """
        SELECT 
            DATE_TRUNC('week', start_time) as week_start,
            COUNT(*) as workouts,
            SUM(duration_seconds)/3600 as hours,
            SUM(distance_meters)/1000 as km,
            AVG(avg_heart_rate) as avg_hr,
            STRING_AGG(DISTINCT activity_type, ', ') as types
        FROM workouts
        WHERE start_time > ?
        GROUP BY week_start
        ORDER BY week_start DESC
        """
        
        results = conn.execute(query, [cutoff]).fetchall()
        conn.close()
        
        # Calculate loads for each week
        loads = self.calculate_training_load(days_back=weeks * 7)
        load_by_week = {}
        for load in loads:
            week = load['day'][:10]  # YYYY-MM-DD
            if week not in load_by_week:
                load_by_week[week] = {'ctl': load['ctl'], 'atl': load['atl'], 'tsb': load['tsb']}
        
        metrics = []
        for row in results:
            week_str = str(row[0])[:10]
            load_data = load_by_week.get(week_str, {'ctl': 0, 'atl': 0, 'tsb': 0})
            
            metrics.append(TrainingMetrics(
                week_start=row[0],
                total_workouts=row[1],
                total_hours=round(row[2], 1) if row[2] else 0,
                total_km=round(row[3], 1) if row[3] else 0,
                avg_heart_rate=int(row[4]) if row[4] else None,
                workout_types=row[5].split(', ') if row[5] else [],
                chronic_load=load_data['ctl'],
                acute_load=load_data['atl'],
                training_stress_balance=load_data['tsb']
            ))
        
        return metrics
    
    def detect_performance_trends(self, activity_type: str = 'run') -> Dict[str, Any]:
        """Detect performance trends for specific activity"""
        conn = duckdb.connect(self.db_path)
        
        query = """
        WITH monthly_stats AS (
            SELECT 
                DATE_TRUNC('month', start_time) as month,
                COUNT(*) as workouts,
                AVG(avg_pace_min_per_km) as avg_pace,
                AVG(avg_heart_rate) as avg_hr,
                AVG(distance_meters/1000) as avg_distance
            FROM workouts
            WHERE activity_type = ?
              AND start_time > CURRENT_TIMESTAMP - INTERVAL '6 months'
              AND avg_pace_min_per_km IS NOT NULL
            GROUP BY month
            ORDER BY month DESC
        )
        SELECT * FROM monthly_stats
        """
        
        results = conn.execute(query, [activity_type]).fetchall()
        conn.close()
        
        if len(results) < 2:
            return {'trend': 'insufficient_data'}
        
        recent = results[0]
        previous = results[1]
        
        # Lower pace = faster (better)
        pace_change = (recent[2] - previous[2]) / previous[2] if previous[2] else 0
        hr_change = (recent[3] - previous[3]) / previous[3] if previous[3] else 0
        
        # Determine trend
        if pace_change < -0.05 and hr_change <= 0:
            trend = 'improving'  # Getting faster at same/lower HR
        elif pace_change > 0.05 and hr_change >= 0:
            trend = 'declining'  # Getting slower at same/higher HR
        else:
            trend = 'stable'
        
        return {
            'activity_type': activity_type,
            'trend': trend,
            'recent_avg_pace': round(recent[2], 2) if recent[2] else None,
            'previous_avg_pace': round(previous[2], 2) if previous[2] else None,
            'pace_change_pct': round(pace_change * 100, 1),
            'recent_avg_hr': int(recent[3]) if recent[3] else None,
            'hr_change_pct': round(hr_change * 100, 1),
            'monthly_workouts': recent[1]
        }
    
    def generate_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive training insights report"""
        print("   🔍 Analyzing training data...")
        
        # Gather all analyses
        overtraining = self.detect_overtraining()
        run_trends = self.detect_performance_trends('run')
        ride_trends = self.detect_performance_trends('ride')
        weekly_patterns = self.analyze_weekly_patterns(weeks=8)
        
        # Calculate current training load
        loads = self.calculate_training_load(days_back=7)
        current_load = loads[0] if loads else None
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'current_training_load': current_load,
            'overtraining_assessment': overtraining,
            'performance_trends': {
                'running': run_trends,
                'cycling': ride_trends
            },
            'recent_weeks': [
                {
                    'week': m.week_start.isoformat() if hasattr(m.week_start, 'isoformat') else str(m.week_start),
                    'workouts': m.total_workouts,
                    'hours': m.total_hours,
                    'km': m.total_km,
                    'avg_hr': m.avg_heart_rate,
                    'types': m.workout_types,
                    'ctl': m.chronic_load,
                    'atl': m.acute_load,
                    'tsb': m.training_stress_balance
                }
                for m in weekly_patterns[:4]
            ]
        }
        
        # Generate narrative summary
        narrative = self._generate_narrative(report)
        report['summary'] = narrative
        
        # Log report to swarm
        log_report(
            agent_id='fitness_analyst',
            realm='fitness',
            report_text=narrative,
            metadata={
                'overtraining_risk': overtraining['overtraining_risk'],
                'current_ctl': current_load['ctl'] if current_load else 0,
                'current_tsb': current_load['tsb'] if current_load else 0
            }
        )
        
        print("   ✅ Report generated")
        return report
    
    def _generate_narrative(self, report: Dict[str, Any]) -> str:
        """Generate human-readable narrative summary"""
        lines = []
        lines.append("📊 TRAINING INSIGHTS REPORT")
        lines.append("=" * 50)
        lines.append("")
        
        # Current load
        current = report['current_training_load']
        if current:
            lines.append(f"Current Training Load:")
            lines.append(f"  • Chronic Load (CTL): {current['ctl']}")
            lines.append(f"  • Acute Load (ATL): {current['atl']}")
            lines.append(f"  • Training Stress Balance: {current['tsb']}")
            lines.append("")
            
            if current['tsb'] > 10:
                lines.append("  ✅ Well-rested, good time for hard sessions")
            elif current['tsb'] < -10:
                lines.append("  ⚠️ Accumulated fatigue, consider recovery")
            else:
                lines.append("  ➡️ Balanced load, maintain current training")
            lines.append("")
        
        # Overtraining
        ot = report['overtraining_assessment']
        if ot.get('overtraining_risk') != 'insufficient_data':
            lines.append(f"Overtraining Risk: {ot['overtraining_risk'].upper()}")
            lines.append(f"  {ot['recommendation']}")
            if ot.get('risk_factors'):
                lines.append(f"  Factors: {', '.join(ot['risk_factors'])}")
            lines.append("")
        
        # Performance trends
        lines.append("Performance Trends:")
        for activity, trend_data in report['performance_trends'].items():
            if trend_data.get('trend') != 'insufficient_data':
                trend = trend_data['trend']
                emoji = '📈' if trend == 'improving' else '📉' if trend == 'declining' else '➡️'
                lines.append(f"  {emoji} {activity.title()}: {trend}")
                if trend_data.get('pace_change_pct'):
                    lines.append(f"     Pace change: {trend_data['pace_change_pct']:+.1f}%")
        lines.append("")
        
        # Recent weeks summary
        lines.append("Last 4 Weeks:")
        for week in report['recent_weeks']:
            lines.append(f"  Week of {week['week'][:10]}:")
            lines.append(f"    {week['workouts']} workouts | {week['hours']}h | {week['km']}km")
            lines.append(f"    TSB: {week['tsb']:+.1f}")
        
        return "\n".join(lines)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fitness Analyst Agent')
    parser.add_argument('command', choices=['report', 'overtraining', 'trends', 'load'], 
                       help='Analysis command')
    parser.add_argument('--db', default='evidence.duckdb', help='Database path')
    parser.add_argument('--activity', default='run', help='Activity type for trends')
    parser.add_argument('--days', type=int, default=30, help='Days to analyze')
    
    args = parser.parse_args()
    
    analyst = FitnessAnalyst(db_path=args.db)
    
    if args.command == 'report':
        report = analyst.generate_insights_report()
        print("\n" + report['summary'])
        print(f"\n💾 Full report saved to events table")
    
    elif args.command == 'overtraining':
        result = analyst.detect_overtraining()
        print(f"\n🔍 Overtraining Assessment:")
        print(json.dumps(result, indent=2))
    
    elif args.command == 'trends':
        result = analyst.detect_performance_trends(args.activity)
        print(f"\n📈 Performance Trends ({args.activity}):")
        print(json.dumps(result, indent=2))
    
    elif args.command == 'load':
        loads = analyst.calculate_training_load(days_back=args.days)
        print(f"\n⚡ Training Load (last {args.days} days):")
        for load in loads[:14]:  # Show last 2 weeks
            print(f"  {load['day']}: CTL={load['ctl']} ATL={load['atl']} TSB={load['tsb']:+.1f}")


if __name__ == '__main__':
    main()
