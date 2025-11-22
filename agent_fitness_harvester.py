#!/usr/bin/env python3
"""
Fitness Data Harvester
Ingests workout telemetry from Strava, Garmin, Fitbit, Apple Health
"""

import os
import json
import time
import duckdb
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import requests
from agent_sdk import log_event

@dataclass
class Workout:
    """Single workout session"""
    workout_id: str
    source: str  # strava, garmin, fitbit, apple_health
    activity_type: str  # run, ride, swim, strength, yoga
    start_time: datetime
    duration_seconds: int
    distance_meters: Optional[float]
    elevation_gain_meters: Optional[float]
    avg_heart_rate: Optional[int]
    max_heart_rate: Optional[int]
    calories: Optional[int]
    avg_power_watts: Optional[int]
    avg_pace_min_per_km: Optional[float]
    moving_time_seconds: Optional[int]
    raw_data: Dict[str, Any]


class FitnessHarvester:
    """Harvest workout data from multiple fitness platforms"""
    
    def __init__(self, db_path: str = 'evidence.duckdb'):
        self.db_path = db_path
        self.init_db()
        
        # Load API credentials from environment
        self.strava_token = os.getenv('STRAVA_ACCESS_TOKEN')
        self.garmin_username = os.getenv('GARMIN_USERNAME')
        self.garmin_password = os.getenv('GARMIN_PASSWORD')
        self.fitbit_token = os.getenv('FITBIT_ACCESS_TOKEN')
        
        print("🏃 Fitness Harvester Initialized")
        
    def init_db(self):
        """Create workout tracking tables"""
        conn = duckdb.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workouts (
                workout_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                activity_type VARCHAR NOT NULL,
                start_time TIMESTAMP NOT NULL,
                duration_seconds INTEGER,
                distance_meters DOUBLE,
                elevation_gain_meters DOUBLE,
                avg_heart_rate INTEGER,
                max_heart_rate INTEGER,
                calories INTEGER,
                avg_power_watts INTEGER,
                avg_pace_min_per_km DOUBLE,
                moving_time_seconds INTEGER,
                raw_data JSON,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workout_streams (
                workout_id VARCHAR NOT NULL,
                stream_type VARCHAR NOT NULL,  -- time, latlng, heartrate, watts, cadence, velocity_smooth
                data_points JSON NOT NULL,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workout_id, stream_type)
            );
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fitness_sync_log (
                sync_id VARCHAR PRIMARY KEY,
                source VARCHAR NOT NULL,
                sync_started TIMESTAMP NOT NULL,
                sync_completed TIMESTAMP,
                workouts_fetched INTEGER DEFAULT 0,
                workouts_new INTEGER DEFAULT 0,
                status VARCHAR,  -- running, completed, failed
                error_message TEXT
            );
        """)
        
        conn.close()
        print("   ✅ Workout schema initialized")
    
    def harvest_strava(self, days_back: int = 30, per_page: int = 50) -> List[Workout]:
        """Fetch workouts from Strava API"""
        if not self.strava_token:
            print("   ⚠️  STRAVA_ACCESS_TOKEN not set, skipping")
            return []
        
        print(f"   🔗 Connecting to Strava API (last {days_back} days)...")
        
        workouts = []
        after = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        headers = {'Authorization': f'Bearer {self.strava_token}'}
        url = 'https://www.strava.com/api/v3/athlete/activities'
        
        try:
            response = requests.get(
                url,
                headers=headers,
                params={'after': after, 'per_page': per_page},
                timeout=30
            )
            response.raise_for_status()
            activities = response.json()
            
            print(f"   📥 Fetched {len(activities)} Strava activities")
            
            for activity in activities:
                workout = Workout(
                    workout_id=f"strava_{activity['id']}",
                    source='strava',
                    activity_type=activity.get('type', 'unknown').lower(),
                    start_time=datetime.fromisoformat(activity['start_date'].replace('Z', '+00:00')),
                    duration_seconds=activity.get('elapsed_time'),
                    distance_meters=activity.get('distance'),
                    elevation_gain_meters=activity.get('total_elevation_gain'),
                    avg_heart_rate=activity.get('average_heartrate'),
                    max_heart_rate=activity.get('max_heartrate'),
                    calories=activity.get('calories'),
                    avg_power_watts=activity.get('average_watts'),
                    avg_pace_min_per_km=self._calc_pace(activity.get('distance'), activity.get('moving_time')),
                    moving_time_seconds=activity.get('moving_time'),
                    raw_data=activity
                )
                workouts.append(workout)
            
            # Optionally fetch detailed streams (HR, power, GPS)
            for workout in workouts[:10]:  # Limit to first 10 to avoid rate limits
                self._fetch_strava_streams(workout)
            
            return workouts
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Strava API error: {e}")
            return []
    
    def _fetch_strava_streams(self, workout: Workout):
        """Fetch detailed time-series data for a workout"""
        if not self.strava_token:
            return
        
        activity_id = workout.workout_id.replace('strava_', '')
        headers = {'Authorization': f'Bearer {self.strava_token}'}
        url = f'https://www.strava.com/api/v3/activities/{activity_id}/streams'
        
        try:
            response = requests.get(
                url,
                headers=headers,
                params={'keys': 'time,heartrate,watts,cadence,velocity_smooth,latlng', 'key_by_type': 'true'},
                timeout=30
            )
            response.raise_for_status()
            streams = response.json()
            
            # Store each stream type
            conn = duckdb.connect(self.db_path)
            for stream_type, stream_data in streams.items():
                conn.execute("""
                    INSERT OR REPLACE INTO workout_streams VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, [workout.workout_id, stream_type, json.dumps(stream_data)])
            conn.close()
            
        except requests.exceptions.RequestException:
            pass  # Stream fetch is optional
    
    def harvest_garmin(self, days_back: int = 30) -> List[Workout]:
        """Fetch workouts from Garmin Connect (requires garth library)"""
        if not (self.garmin_username and self.garmin_password):
            print("   ⚠️  GARMIN credentials not set, skipping")
            return []
        
        print(f"   🔗 Connecting to Garmin Connect (last {days_back} days)...")
        
        try:
            from garth.exc import GarthHTTPError
            import garth
            
            # Authenticate
            try:
                garth.resume('~/.garth')
            except:
                garth.login(self.garmin_username, self.garmin_password)
                garth.save('~/.garth')
            
            # Fetch activities
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            activities = garth.list_activities(start_date, end_date)
            print(f"   📥 Fetched {len(activities)} Garmin activities")
            
            workouts = []
            for activity in activities:
                workout = Workout(
                    workout_id=f"garmin_{activity['activityId']}",
                    source='garmin',
                    activity_type=activity.get('activityType', {}).get('typeKey', 'unknown').lower(),
                    start_time=datetime.fromisoformat(activity['startTimeLocal']),
                    duration_seconds=int(activity.get('duration', 0)),
                    distance_meters=activity.get('distance'),
                    elevation_gain_meters=activity.get('elevationGain'),
                    avg_heart_rate=activity.get('averageHR'),
                    max_heart_rate=activity.get('maxHR'),
                    calories=activity.get('calories'),
                    avg_power_watts=activity.get('avgPower'),
                    avg_pace_min_per_km=self._calc_pace(activity.get('distance'), activity.get('movingDuration')),
                    moving_time_seconds=int(activity.get('movingDuration', 0)),
                    raw_data=activity
                )
                workouts.append(workout)
            
            return workouts
            
        except ImportError:
            print("   ⚠️  garth library not installed (pip install garth)")
            return []
        except Exception as e:
            print(f"   ❌ Garmin error: {e}")
            return []
    
    def harvest_fitbit(self, days_back: int = 30) -> List[Workout]:
        """Fetch workouts from Fitbit API"""
        if not self.fitbit_token:
            print("   ⚠️  FITBIT_ACCESS_TOKEN not set, skipping")
            return []
        
        print(f"   🔗 Connecting to Fitbit API (last {days_back} days)...")
        
        workouts = []
        headers = {'Authorization': f'Bearer {self.fitbit_token}'}
        
        try:
            # Fitbit requires date range queries
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f'https://api.fitbit.com/1/user/-/activities/list.json'
            response = requests.get(
                url,
                headers=headers,
                params={
                    'afterDate': start_date.strftime('%Y-%m-%d'),
                    'sort': 'desc',
                    'limit': 100,
                    'offset': 0
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            activities = data.get('activities', [])
            print(f"   📥 Fetched {len(activities)} Fitbit activities")
            
            for activity in activities:
                workout = Workout(
                    workout_id=f"fitbit_{activity['logId']}",
                    source='fitbit',
                    activity_type=activity.get('activityName', 'unknown').lower(),
                    start_time=datetime.fromisoformat(activity['startTime']),
                    duration_seconds=activity.get('duration', 0) // 1000,  # Fitbit uses milliseconds
                    distance_meters=activity.get('distance', 0) * 1000,  # Fitbit uses km
                    elevation_gain_meters=activity.get('elevationGain'),
                    avg_heart_rate=activity.get('averageHeartRate'),
                    max_heart_rate=None,
                    calories=activity.get('calories'),
                    avg_power_watts=None,
                    avg_pace_min_per_km=activity.get('pace'),
                    moving_time_seconds=activity.get('activeDuration', 0) // 1000,
                    raw_data=activity
                )
                workouts.append(workout)
            
            return workouts
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Fitbit API error: {e}")
            return []
    
    def harvest_apple_health(self, export_path: str = None) -> List[Workout]:
        """Parse workouts from Apple Health export.xml"""
        if not export_path:
            export_path = os.path.expanduser('~/Downloads/apple_health_export/export.xml')
        
        export_file = Path(export_path)
        if not export_file.exists():
            print(f"   ⚠️  Apple Health export not found at {export_path}")
            return []
        
        print(f"   📂 Parsing Apple Health export...")
        
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(export_file)
            root = tree.getroot()
            
            workouts = []
            workout_elements = root.findall('.//Workout')
            
            print(f"   📥 Found {len(workout_elements)} Apple Health workouts")
            
            for workout_elem in workout_elements:
                workout_type = workout_elem.get('workoutActivityType', 'HKWorkoutActivityTypeOther')
                start_date = datetime.fromisoformat(workout_elem.get('startDate'))
                end_date = datetime.fromisoformat(workout_elem.get('endDate'))
                duration = (end_date - start_date).total_seconds()
                
                # Parse optional metrics
                distance = float(workout_elem.get('totalDistance', 0))
                energy = float(workout_elem.get('totalEnergyBurned', 0))
                
                workout = Workout(
                    workout_id=f"apple_{workout_elem.get('startDate')}",
                    source='apple_health',
                    activity_type=workout_type.replace('HKWorkoutActivityType', '').lower(),
                    start_time=start_date,
                    duration_seconds=int(duration),
                    distance_meters=distance * 1000 if distance else None,
                    elevation_gain_meters=None,
                    avg_heart_rate=None,
                    max_heart_rate=None,
                    calories=int(energy) if energy else None,
                    avg_power_watts=None,
                    avg_pace_min_per_km=None,
                    moving_time_seconds=int(duration),
                    raw_data=workout_elem.attrib
                )
                workouts.append(workout)
            
            return workouts
            
        except Exception as e:
            print(f"   ❌ Apple Health parsing error: {e}")
            return []
    
    def _calc_pace(self, distance_m: Optional[float], time_s: Optional[int]) -> Optional[float]:
        """Calculate pace in min/km"""
        if not (distance_m and time_s and distance_m > 0):
            return None
        km = distance_m / 1000
        minutes = time_s / 60
        return minutes / km
    
    def store_workouts(self, workouts: List[Workout]) -> int:
        """Store workouts in database, skip duplicates"""
        if not workouts:
            return 0
        
        conn = duckdb.connect(self.db_path)
        new_count = 0
        
        for workout in workouts:
            # Check if workout already exists
            existing = conn.execute(
                "SELECT workout_id FROM workouts WHERE workout_id = ?",
                [workout.workout_id]
            ).fetchone()
            
            if existing:
                continue
            
            # Insert new workout
            conn.execute("""
                INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                workout.workout_id,
                workout.source,
                workout.activity_type,
                workout.start_time,
                workout.duration_seconds,
                workout.distance_meters,
                workout.elevation_gain_meters,
                workout.avg_heart_rate,
                workout.max_heart_rate,
                workout.calories,
                workout.avg_power_watts,
                workout.avg_pace_min_per_km,
                workout.moving_time_seconds,
                json.dumps(workout.raw_data)
            ])
            new_count += 1
        
        conn.close()
        print(f"   ✅ Stored {new_count} new workouts ({len(workouts) - new_count} duplicates skipped)")
        return new_count
    
    def sync_all(self, days_back: int = 30) -> Dict[str, int]:
        """Harvest from all sources and store"""
        sync_id = f"sync_{int(time.time())}"
        
        # Log sync start
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            INSERT INTO fitness_sync_log VALUES (?, 'all', CURRENT_TIMESTAMP, NULL, 0, 0, 'running', NULL)
        """, [sync_id])
        conn.close()
        
        print(f"\n🔄 Starting fitness sync (sync_id: {sync_id})")
        
        all_workouts = []
        sources_synced = []
        
        # Harvest from each source
        try:
            strava_workouts = self.harvest_strava(days_back=days_back)
            all_workouts.extend(strava_workouts)
            if strava_workouts:
                sources_synced.append('strava')
        except Exception as e:
            print(f"   ⚠️  Strava sync failed: {e}")
        
        try:
            garmin_workouts = self.harvest_garmin(days_back=days_back)
            all_workouts.extend(garmin_workouts)
            if garmin_workouts:
                sources_synced.append('garmin')
        except Exception as e:
            print(f"   ⚠️  Garmin sync failed: {e}")
        
        try:
            fitbit_workouts = self.harvest_fitbit(days_back=days_back)
            all_workouts.extend(fitbit_workouts)
            if fitbit_workouts:
                sources_synced.append('fitbit')
        except Exception as e:
            print(f"   ⚠️  Fitbit sync failed: {e}")
        
        # Store workouts
        new_count = self.store_workouts(all_workouts)
        
        # Log completion
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            UPDATE fitness_sync_log 
            SET sync_completed = CURRENT_TIMESTAMP,
                workouts_fetched = ?,
                workouts_new = ?,
                status = 'completed'
            WHERE sync_id = ?
        """, [len(all_workouts), new_count, sync_id])
        conn.close()
        
        # Log event to swarm
        log_event(
            event_type='fitness_sync_completed',
            agent_id='fitness_harvester',
            realm='fitness',
            metadata={
                'sync_id': sync_id,
                'sources': sources_synced,
                'workouts_fetched': len(all_workouts),
                'workouts_new': new_count
            },
            content=f"Synced {len(all_workouts)} workouts from {', '.join(sources_synced)}"
        )
        
        print(f"\n✅ Sync complete: {len(all_workouts)} fetched, {new_count} new")
        print(f"   Sources: {', '.join(sources_synced)}")
        
        return {
            'sync_id': sync_id,
            'total_fetched': len(all_workouts),
            'total_new': new_count,
            'sources': sources_synced
        }
    
    def get_workout_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get workout summary for last N days"""
        conn = duckdb.connect(self.db_path)
        
        cutoff = datetime.now() - timedelta(days=days)
        
        result = conn.execute("""
            SELECT 
                COUNT(*) as total_workouts,
                COUNT(DISTINCT activity_type) as activity_types,
                SUM(distance_meters)/1000 as total_km,
                SUM(duration_seconds)/3600 as total_hours,
                SUM(calories) as total_calories,
                AVG(avg_heart_rate) as avg_hr,
                source
            FROM workouts
            WHERE start_time > ?
            GROUP BY source
        """, [cutoff]).fetchall()
        
        conn.close()
        
        summary = {
            'period_days': days,
            'by_source': {}
        }
        
        for row in result:
            summary['by_source'][row[6]] = {
                'workouts': row[0],
                'activity_types': row[1],
                'total_km': round(row[2], 1) if row[2] else 0,
                'total_hours': round(row[3], 1) if row[3] else 0,
                'total_calories': int(row[4]) if row[4] else 0,
                'avg_hr': int(row[5]) if row[5] else None
            }
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fitness Data Harvester')
    parser.add_argument('command', choices=['sync', 'summary', 'test'], help='Command to run')
    parser.add_argument('--days', type=int, default=30, help='Days to look back (default: 30)')
    parser.add_argument('--db', default='evidence.duckdb', help='Database path')
    
    args = parser.parse_args()
    
    harvester = FitnessHarvester(db_path=args.db)
    
    if args.command == 'sync':
        result = harvester.sync_all(days_back=args.days)
        print(f"\n📊 Result: {result}")
    
    elif args.command == 'summary':
        summary = harvester.get_workout_summary(days=args.days)
        print(f"\n📊 Workout Summary (last {args.days} days):")
        print(json.dumps(summary, indent=2))
    
    elif args.command == 'test':
        print("\n🧪 Testing connections...")
        print(f"   Strava token: {'✅ Set' if harvester.strava_token else '❌ Missing'}")
        print(f"   Garmin creds: {'✅ Set' if (harvester.garmin_username and harvester.garmin_password) else '❌ Missing'}")
        print(f"   Fitbit token: {'✅ Set' if harvester.fitbit_token else '❌ Missing'}")


if __name__ == '__main__':
    main()
