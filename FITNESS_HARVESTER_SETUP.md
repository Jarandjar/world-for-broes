# Fitness Data Harvester Setup

## Overview
Ingests workout telemetry from Strava, Garmin Connect, Fitbit, and Apple Health into the swarm's `evidence.duckdb`.

## Quick Start

```bash
# Install dependencies
pip install requests garth

# Set API tokens (add to .env)
export STRAVA_ACCESS_TOKEN="your_token_here"
export GARMIN_USERNAME="your_email"
export GARMIN_PASSWORD="your_password"
export FITBIT_ACCESS_TOKEN="your_token_here"

# Test connections
python agent_fitness_harvester.py test

# Sync last 30 days
python agent_fitness_harvester.py sync --days 30

# View summary
python agent_fitness_harvester.py summary --days 7
```

## API Setup Guides

### Strava
1. Create app at https://www.strava.com/settings/api
2. Get access token: https://developers.strava.com/docs/getting-started/#oauth
3. Set `STRAVA_ACCESS_TOKEN` environment variable

**Quick OAuth Flow:**
```python
import requests

# Exchange code for token
response = requests.post(
    'https://www.strava.com/oauth/token',
    data={
        'client_id': 'YOUR_CLIENT_ID',
        'client_secret': 'YOUR_CLIENT_SECRET',
        'code': 'AUTHORIZATION_CODE',
        'grant_type': 'authorization_code'
    }
)
token = response.json()['access_token']
```

### Garmin Connect
Uses the `garth` library for authentication (handles 2FA).

```bash
pip install garth
```

First run will prompt for login and save credentials to `~/.garth`.

### Fitbit
1. Create app at https://dev.fitbit.com/apps/new
2. Get OAuth 2.0 token
3. Set `FITBIT_ACCESS_TOKEN`

### Apple Health
1. Export data from iPhone: Health app → Profile → Export All Health Data
2. Extract `export.xml` to `~/Downloads/apple_health_export/`
3. Run sync (auto-detects export file)

## Database Schema

### workouts
```sql
workout_id VARCHAR PRIMARY KEY         -- "strava_12345", "garmin_67890"
source VARCHAR                         -- strava, garmin, fitbit, apple_health
activity_type VARCHAR                  -- run, ride, swim, strength
start_time TIMESTAMP
duration_seconds INTEGER
distance_meters DOUBLE
elevation_gain_meters DOUBLE
avg_heart_rate INTEGER
max_heart_rate INTEGER
calories INTEGER
avg_power_watts INTEGER
avg_pace_min_per_km DOUBLE
moving_time_seconds INTEGER
raw_data JSON                          -- Full API response
ingested_at TIMESTAMP
```

### workout_streams
```sql
workout_id VARCHAR
stream_type VARCHAR                    -- time, latlng, heartrate, watts, cadence
data_points JSON                       -- Time-series data
```

### fitness_sync_log
```sql
sync_id VARCHAR
source VARCHAR
sync_started TIMESTAMP
sync_completed TIMESTAMP
workouts_fetched INTEGER
workouts_new INTEGER
status VARCHAR                         -- running, completed, failed
error_message TEXT
```

## Usage Examples

### Manual Sync
```python
from agent_fitness_harvester import FitnessHarvester

harvester = FitnessHarvester()
result = harvester.sync_all(days_back=30)
# {'sync_id': 'sync_1234567890', 'total_fetched': 42, 'total_new': 12, 'sources': ['strava', 'garmin']}
```

### Query Workouts
```python
import duckdb

conn = duckdb.connect('evidence.duckdb')

# Total distance by activity type
conn.execute("""
    SELECT activity_type, SUM(distance_meters)/1000 as total_km
    FROM workouts
    WHERE start_time > CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY activity_type
    ORDER BY total_km DESC
""").fetchall()

# Average heart rate trends
conn.execute("""
    SELECT DATE_TRUNC('day', start_time) as day, 
           AVG(avg_heart_rate) as avg_hr
    FROM workouts
    WHERE avg_heart_rate IS NOT NULL
    GROUP BY day
    ORDER BY day DESC
    LIMIT 30
""").fetchall()
```

### Weekly Summary
```python
harvester = FitnessHarvester()
summary = harvester.get_workout_summary(days=7)
print(summary)
# {
#   'period_days': 7,
#   'by_source': {
#     'strava': {'workouts': 5, 'total_km': 42.3, 'total_hours': 4.2, 'total_calories': 2100, 'avg_hr': 145},
#     'garmin': {'workouts': 2, 'total_km': 18.5, 'total_hours': 1.8, 'total_calories': 950, 'avg_hr': 138}
#   }
# }
```

## Automated Sync

Add to `swarm_orchestrator.py` task definitions:

```python
# Weekly fitness sync
self.governor.create_job(
    JobType.FITNESS_SYNC,
    meta={'days_back': 7, 'sources': ['strava', 'garmin']},
    priority=5
)
```

Or schedule via cron:
```bash
# Sync every morning at 6am
0 6 * * * cd /path/to/Science && python agent_fitness_harvester.py sync --days 1
```

## Data Privacy Notes

- Tokens stored in environment variables (not committed to git)
- Garmin credentials cached in `~/.garth` (use keyring for production)
- Raw API responses stored in `raw_data` JSON field
- No PII transmitted outside local database
- Apple Health export contains full health history (secure accordingly)

## Rate Limits

| Source | Limit | Notes |
|--------|-------|-------|
| Strava | 100 req/15min, 1000 req/day | Detailed streams count separately |
| Garmin | None documented | Respectful usage recommended |
| Fitbit | 150 req/hour | Personal use limits |
| Apple Health | N/A | Local file parse |

## Troubleshooting

### Strava 401 Unauthorized
- Token expired (refresh via OAuth)
- Check token has `activity:read_all` scope

### Garmin Login Fails
```bash
# Clear cached credentials
rm -rf ~/.garth
# Re-run sync (will prompt for login)
```

### Fitbit Token Refresh
Fitbit tokens expire after 8 hours. Implement refresh flow:
```python
response = requests.post(
    'https://api.fitbit.com/oauth2/token',
    data={
        'grant_type': 'refresh_token',
        'refresh_token': 'YOUR_REFRESH_TOKEN'
    },
    headers={'Authorization': f'Basic {base64_encoded_client_credentials}'}
)
```

## Advanced: Workout Analysis

### Analyze Training Load
```python
# Calculate chronic training load (CTL) - 42-day rolling average
conn.execute("""
    SELECT start_time::DATE as day,
           AVG(duration_seconds * avg_heart_rate / 3600.0) 
               OVER (ORDER BY start_time::DATE ROWS BETWEEN 41 PRECEDING AND CURRENT ROW) as ctl
    FROM workouts
    WHERE activity_type IN ('run', 'ride')
    ORDER BY day DESC
""").fetchall()
```

### Detect Overtraining
```python
# Find weeks with high volume + low avg HR (potential fatigue)
conn.execute("""
    SELECT DATE_TRUNC('week', start_time) as week,
           COUNT(*) as workouts,
           SUM(duration_seconds)/3600 as hours,
           AVG(avg_heart_rate) as avg_hr
    FROM workouts
    GROUP BY week
    HAVING hours > 10 AND avg_hr < (SELECT AVG(avg_heart_rate) * 0.9 FROM workouts)
""").fetchall()
```

## Next Steps

1. **Add more sources**: Wahoo, Peloton, Zwift APIs
2. **Real-time sync**: Webhooks for instant updates
3. **Agent analysis**: Create `OracleAgent` job type for workout insights
4. **Visualization**: Generate training load charts via `AlchemistAgent`
5. **Anomaly detection**: Detect unusual HR patterns or performance drops
