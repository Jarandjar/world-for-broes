#!/usr/bin/env python3
"""
Fitness Dashboard Generator
Creates HTML dashboard with training charts using plotly
"""

import duckdb
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  plotly not installed. Run: pip install plotly")


class FitnessDashboard:
    """Generate interactive fitness dashboards"""
    
    def __init__(self, db_path: str = 'evidence.duckdb'):
        self.db_path = db_path
        self.output_dir = Path('out/fitness_dashboard')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dashboard(self, days: int = 90) -> str:
        """Generate complete fitness dashboard HTML"""
        if not PLOTLY_AVAILABLE:
            return "ERROR: plotly not installed"
        
        conn = duckdb.connect(self.db_path)
        cutoff = datetime.now() - timedelta(days=days)
        
        print(f"📊 Generating fitness dashboard (last {days} days)...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Weekly Volume (hours)', 
                'Training Load (CTL/ATL/TSB)',
                'Heart Rate Trends',
                'Distance by Activity',
                'Workout Frequency',
                'Performance Trends (Pace)'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Weekly Volume
        weekly_data = conn.execute("""
            SELECT DATE_TRUNC('week', start_time) as week,
                   SUM(duration_seconds)/3600 as hours
            FROM workouts
            WHERE start_time > ?
            GROUP BY week
            ORDER BY week
        """, [cutoff]).fetchall()
        
        if weekly_data:
            weeks = [row[0] for row in weekly_data]
            hours = [row[1] for row in weekly_data]
            fig.add_trace(
                go.Bar(x=weeks, y=hours, name='Hours', marker_color='lightblue'),
                row=1, col=1
            )
        
        # 2. Training Load (CTL/ATL/TSB)
        load_data = conn.execute("""
            WITH daily_load AS (
                SELECT 
                    start_time::DATE as day,
                    SUM(duration_seconds * COALESCE(avg_heart_rate, 140) / 3600.0) as load
                FROM workouts
                WHERE start_time > ?
                GROUP BY day
            ),
            ctl_calc AS (
                SELECT 
                    day,
                    AVG(load) OVER (ORDER BY day ROWS BETWEEN 41 PRECEDING AND CURRENT ROW) as ctl,
                    AVG(load) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as atl
                FROM daily_load
            )
            SELECT day, ctl, atl, ctl - atl as tsb
            FROM ctl_calc
            ORDER BY day
        """, [cutoff]).fetchall()
        
        if load_data:
            days_list = [row[0] for row in load_data]
            ctl = [row[1] for row in load_data]
            atl = [row[2] for row in load_data]
            tsb = [row[3] for row in load_data]
            
            fig.add_trace(go.Scatter(x=days_list, y=ctl, name='CTL (Chronic)', line=dict(color='blue')), row=1, col=2)
            fig.add_trace(go.Scatter(x=days_list, y=atl, name='ATL (Acute)', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=days_list, y=tsb, name='TSB (Balance)', line=dict(color='green', dash='dash')), row=1, col=2)
        
        # 3. Heart Rate Trends
        hr_data = conn.execute("""
            SELECT DATE_TRUNC('week', start_time) as week,
                   AVG(avg_heart_rate) as avg_hr,
                   AVG(max_heart_rate) as max_hr
            FROM workouts
            WHERE start_time > ? AND avg_heart_rate IS NOT NULL
            GROUP BY week
            ORDER BY week
        """, [cutoff]).fetchall()
        
        if hr_data:
            weeks_hr = [row[0] for row in hr_data]
            avg_hr = [row[1] for row in hr_data]
            max_hr = [row[2] for row in hr_data]
            
            fig.add_trace(go.Scatter(x=weeks_hr, y=avg_hr, name='Avg HR', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=weeks_hr, y=max_hr, name='Max HR', line=dict(color='red', dash='dot')), row=2, col=1)
        
        # 4. Distance by Activity
        activity_data = conn.execute("""
            SELECT activity_type, SUM(distance_meters)/1000 as total_km
            FROM workouts
            WHERE start_time > ? AND distance_meters IS NOT NULL
            GROUP BY activity_type
            ORDER BY total_km DESC
        """, [cutoff]).fetchall()
        
        if activity_data:
            activities = [row[0] for row in activity_data]
            distances = [row[1] for row in activity_data]
            fig.add_trace(
                go.Bar(x=activities, y=distances, name='Distance (km)', marker_color='lightgreen'),
                row=2, col=2
            )
        
        # 5. Workout Frequency
        freq_data = conn.execute("""
            SELECT DATE_TRUNC('week', start_time) as week,
                   COUNT(*) as workout_count
            FROM workouts
            WHERE start_time > ?
            GROUP BY week
            ORDER BY week
        """, [cutoff]).fetchall()
        
        if freq_data:
            weeks_freq = [row[0] for row in freq_data]
            counts = [row[1] for row in freq_data]
            fig.add_trace(
                go.Bar(x=weeks_freq, y=counts, name='Workouts', marker_color='purple'),
                row=3, col=1
            )
        
        # 6. Performance Trends (Pace for runs)
        pace_data = conn.execute("""
            SELECT DATE_TRUNC('week', start_time) as week,
                   AVG(avg_pace_min_per_km) as avg_pace
            FROM workouts
            WHERE start_time > ? 
              AND activity_type = 'run'
              AND avg_pace_min_per_km IS NOT NULL
              AND avg_pace_min_per_km < 10  -- Filter outliers
            GROUP BY week
            ORDER BY week
        """, [cutoff]).fetchall()
        
        if pace_data:
            weeks_pace = [row[0] for row in pace_data]
            paces = [row[1] for row in pace_data]
            fig.add_trace(
                go.Scatter(x=weeks_pace, y=paces, name='Pace (min/km)', 
                          line=dict(color='teal'), mode='lines+markers'),
                row=3, col=2
            )
        
        conn.close()
        
        # Update layout
        fig.update_layout(
            title_text=f"Fitness Dashboard - Last {days} Days",
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Add axis labels
        fig.update_yaxes(title_text="Hours", row=1, col=1)
        fig.update_yaxes(title_text="Load", row=1, col=2)
        fig.update_yaxes(title_text="BPM", row=2, col=1)
        fig.update_yaxes(title_text="Kilometers", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_yaxes(title_text="Min/km", row=3, col=2)
        
        # Save HTML
        output_file = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_file))
        
        print(f"✅ Dashboard saved: {output_file}")
        return str(output_file)
    
    def generate_summary_stats(self, days: int = 30) -> dict:
        """Generate summary statistics for dashboard"""
        conn = duckdb.connect(self.db_path)
        cutoff = datetime.now() - timedelta(days=days)
        
        stats = {}
        
        # Total workouts
        result = conn.execute("""
            SELECT COUNT(*), 
                   SUM(duration_seconds)/3600,
                   SUM(distance_meters)/1000,
                   AVG(avg_heart_rate)
            FROM workouts
            WHERE start_time > ?
        """, [cutoff]).fetchone()
        
        stats['total_workouts'] = result[0] if result[0] else 0
        stats['total_hours'] = round(result[1], 1) if result[1] else 0
        stats['total_km'] = round(result[2], 1) if result[2] else 0
        stats['avg_hr'] = int(result[3]) if result[3] else None
        
        # Current training load
        load = conn.execute("""
            WITH daily_load AS (
                SELECT 
                    start_time::DATE as day,
                    SUM(duration_seconds * COALESCE(avg_heart_rate, 140) / 3600.0) as load
                FROM workouts
                WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '90 days'
                GROUP BY day
            ),
            ctl_calc AS (
                SELECT 
                    day,
                    AVG(load) OVER (ORDER BY day ROWS BETWEEN 41 PRECEDING AND CURRENT ROW) as ctl,
                    AVG(load) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as atl
                FROM daily_load
            )
            SELECT ctl, atl, ctl - atl as tsb
            FROM ctl_calc
            ORDER BY day DESC
            LIMIT 1
        """).fetchone()
        
        if load:
            stats['ctl'] = round(load[0], 1) if load[0] else 0
            stats['atl'] = round(load[1], 1) if load[1] else 0
            stats['tsb'] = round(load[2], 1) if load[2] else 0
        
        conn.close()
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fitness Dashboard Generator')
    parser.add_argument('--days', type=int, default=90, help='Days to include')
    parser.add_argument('--db', default='evidence.duckdb', help='Database path')
    parser.add_argument('--stats', action='store_true', help='Print summary stats only')
    
    args = parser.parse_args()
    
    dashboard = FitnessDashboard(db_path=args.db)
    
    if args.stats:
        stats = dashboard.generate_summary_stats(days=args.days)
        print(f"\n📊 Summary Stats (last {args.days} days):")
        print(json.dumps(stats, indent=2))
    else:
        output_file = dashboard.generate_dashboard(days=args.days)
        print(f"\n🌐 Open in browser: file://{output_file}")


if __name__ == '__main__':
    main()
