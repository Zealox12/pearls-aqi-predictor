import sys
import types

m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime,timedelta
import hopsworks
import pandas as pd
import time

API_KEY = None
API_KEY_HS = None

def fetch_historical_air_pollution(lat, lon, start_ts, end_ts):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        'lat': lat,
        'lon': lon,
        'start': int(start_ts),
        'end': int(end_ts),
        'appid': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json().get('list', [])
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None

def store_features(fs, df):
    if df is None or df.empty:
        print("No data to store")
        return None

    fg = fs.get_or_create_feature_group(name="weather_pollution_features",
    version=1,
    description="Weather and air pollution features for Karachi",
    primary_key=["location", "event_timestamp"],
    event_time="event_timestamp",
    online_enabled=True,
    )

    if fg is not None:
        fg.insert(df, wait = True)
        print("Feature group 'weather_pollution_features' is ready")
    else:
        print("Failed to create or retrieve feature group 'weather_pollution_features'")

    return fg

def build_historical_record(hour_data, location):
    components = hour_data.get('components', {})
    timestamp = hour_data.get('dt')
    event_ts = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

    record = {
        'temperature': 0.0,
        'feels_like': 0.0,
        'temp_min': 0.0,
        'temp_max': 0.0,
        'humidity': 0,
        'pressure': 0.0,
        'sea_level': 0.0,
        'grnd_level': 0.0,
        'wind_speed': 0.0,
        'wind_direction': 0,
        'wind_gust': 0.0,
        'clouds': 0,
        'weather_main': 'unknown',
        'weather_description': 'unknown',
        'rain_1h': 0,
        'snow_1h': 0,
        'location': str(location),
        'event_timestamp': event_ts,
        'timestamp': int(timestamp) if timestamp is not None else 0,
        'aqi_is_poor': int(1 if hour_data.get('main', {}).get('aqi',0) >= 3 else 0),
        'pm25_exceeds_who': int(1 if components.get('pm2_5', 0) > 15 else 0),
        'aqi': int(hour_data.get('main', {}).get('aqi', 0)),
        'co': float(components.get('co', 0.0)),
        'no': int(components.get('no', 0)) if components.get('no') is not None else 0,
        'no2': float(components.get('no2', 0.0)),
        'o3': float(components.get('o3', 0.0)),
        'so2': float(components.get('so2', 0.0)),
        'pm2_5': float(components.get('pm2_5', 0.0)),
        'pm10': float(components.get('pm10', 0.0)),
        'nh3': int(components.get('nh3', 0)) if components.get('nh3') is not None else 0
    }
    return record

def build_historical_dataframe(hourly_data, location):
    if not hourly_data:
        return None
    
    records = []
    for hour in hourly_data:
        record = build_historical_record(hour, location)
        records.append(record)
    return pd.DataFrame(records)

def backfill_data(fs, location, lat, lon, start_date, end_date):
    print(f"Backfilling data for {location} from {start_date} to {end_date}...")

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    hourly_data = fetch_historical_air_pollution(lat, lon, start_ts, end_ts)

    if hourly_data:
        print(f"Fetched {len(hourly_data)} records for {location}")
        df = build_historical_dataframe(hourly_data, location)

        if df is not None and not df.empty:
            print(f"Storing {len(df)} records for {location} in feature store...")
            store_features(fs, df)
            return df
    return None

def main():
    global API_KEY, API_KEY_HS
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_KEY_HS = os.getenv("API_KEY_HS")

    if not API_KEY or not API_KEY_HS:
        raise ValueError("One or more API keys not found in .env file")
    
    project = hopsworks.login(
        api_key_value=API_KEY_HS,
        project="Pearls_AQI_Predictor12",
        host = "eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    print("Connected to Hopsworks")

    KARACHI_LYARI_LAT = 24.8607
    KARACHI_LYARI_LON = 67.0011

    print("Fetching historical air pollution data for Karachi Lyari...")

    end_date = datetime.now()
    for months_ago in range(1, 69):
        start_date = end_date - timedelta(days=30 * months_ago)
        end_chunk = end_date - timedelta(days=30 * (months_ago - 1))
        backfill_data(fs, "Karachi_Lyari", KARACHI_LYARI_LAT, KARACHI_LYARI_LON, start_date, end_chunk)
        time.sleep(1)

    fg = fs.get_feature_group(name="weather_pollution_features", version=1)
    df = fg.read()
    print(f"Total records in feature store: {len(df)}")
    print(f"location: {df['location'].unique()}")
    print(f"Date range: {df['event_timestamp'].min()} to {df['event_timestamp'].max()}")

if __name__ == "__main__":
    main()