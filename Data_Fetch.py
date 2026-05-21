import sys
import types

m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m
import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import hopsworks
import pandas as pd
import time
API_KEY = None
API_KEY_HS = None


def fetch_current_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY,
        'units': 'metric'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None
    
def fetch_current_air_pollution(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None


def parse_weather_data(weather_data):
    if weather_data is None:
        return None

    parsed_data = {
        'temperature': float(weather_data.get('main', {}).get('temp', 0)),
        'feels_like': float(weather_data.get('main',{}).get('feels_like', 0)),
        'temp_min': float(weather_data.get('main', {}).get('temp_min', 0)),
        'temp_max': float(weather_data.get('main', {}).get('temp_max', 0)),
        'humidity': int(weather_data.get('main', {}).get('humidity',0)),
        'pressure': float(weather_data.get('main', {}).get('pressure', 0)),
        'sea_level': float(weather_data.get('main', {}).get('sea_level', 0)) if weather_data.get('main', {}).get('sea_level') else 0.0,
        'grnd_level': float(weather_data.get('main', {}).get('grnd_level', 0)) if weather_data.get('main', {}).get('grnd_level') else 0.0,
        'wind_speed': float(weather_data.get('wind', {}).get('speed', 0)),
        'wind_direction': int(weather_data.get('wind', {}).get('deg',0)),
        'wind_gust': float(weather_data.get('wind', {}).get('gust',0)) if weather_data.get('wind', {}).get('gust') else 0.0,
        'clouds': int(weather_data.get('clouds', {}).get('all', 0)),
        'weather_main': str(weather_data.get('weather', [{}])[0].get('main', 'unknown')),
        'weather_description': str(weather_data.get('weather', [{}])[0].get('description', 'unknown')),
        'rain_1h': int(weather_data.get('rain', {}).get('1h', 0)),
        'snow_1h': int(weather_data.get('snow', {}).get('1h', 0)),
        'timestamp': int(weather_data.get('dt', 0))
    }
    return parsed_data

def parse_air_pollution_data(air_pollution_data):
    if air_pollution_data is None:
        return None

    components = air_pollution_data.get('list', [{}])[0].get('components', {})
    parsed_data = {
        'co': float(components.get('co', 0.0)),
        'no': int(components.get('no', 0)) if components.get('no') is not None else 0,
        'no2': float(components.get('no2', 0.0)),
        'o3': float(components.get('o3', 0.0)),
        'so2': float(components.get('so2', 0.0)),
        'pm2_5': float(components.get('pm2_5', 0.0)),
        'pm10': float(components.get('pm10', 0.0)),
        'nh3': int(components.get('nh3', 0)) if components.get('nh3') is not None else 0,
        'aqi': int(air_pollution_data.get('list', [{}])[0].get('main', {}).get('aqi', 1)),
        'timestamp': int(air_pollution_data.get('list', [{}])[0].get('dt', 0))
    }
    return parsed_data

def build_record(weather_parsed, pollution_parsed, location_name):
    if weather_parsed is None or pollution_parsed is None:
        return None
    
    record = {**weather_parsed, **pollution_parsed}

    record['location'] = location_name

    if record.get('timestamp'):
        record['event_timestamp'] = datetime.fromtimestamp(record['timestamp'])
    else:
        record['event_timestamp'] = datetime.now()

    record['aqi_is_poor'] = 1 if record.get('aqi') and record['aqi'] >= 3 else 0
    record['pm25_exceeds_who'] = 1 if record.get('pm2_5') and record['pm2_5'] > 15 else 0

    return pd.DataFrame([record])

def store_features(fs, features_df):
    if features_df is None or features_df.empty:
        print("No data to store")
        return None
    
    feature_group = fs.get_or_create_feature_group(
        name="weather_pollution_features",
        version=1,
        description="Weather and air pollution features for Karachi",
        primary_key=['location', 'event_timestamp'],
        event_time='event_timestamp',
        online_enabled=True
    )
    
    if feature_group is not None:
        feature_group.insert(features_df, wait=False)
        print(f"✅ Stored {len(features_df)} record(s) in feature store")
    else:
        print("❌ Could not create feature group")
    
    return feature_group

def is_duplicate_record(fs, location, event_timestamp):
    try:
        fg = fs.get_feature_group(name="weather_pollution_features", version=1)
        df = fg.read()
        if len(df) > 0:
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
            matches = df[df['event_timestamp'] == event_timestamp]
            return len(matches) > 0
        return False
    except Exception as e:
        print(f"Error checking for duplicate record: {e}")
        return False

def main():
    global API_KEY
    global API_KEY_HS
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    API_KEY_HS = os.getenv('API_KEY_HS')
    if not API_KEY:
        raise ValueError("API key not found in .env file")
    if not API_KEY_HS:
        raise ValueError("API_KEY_HS not found in .env file")
    
    project = hopsworks.login(
    api_key_value=API_KEY_HS,
    project="Pearls_AQI_Predictor12",
    host = "eu-west.cloud.hopsworks.ai"
    )

    fs = project.get_feature_store()
    print("Connected to Hopsworks feature store")

    KARACHI_Lyari_LAT = 24.8607
    KARACHI_Lyari_LON = 67.0011

    weather_raw = fetch_current_weather(KARACHI_Lyari_LAT, KARACHI_Lyari_LON)
    time.sleep(0.1)

    pollution_raw = fetch_current_air_pollution(KARACHI_Lyari_LAT, KARACHI_Lyari_LON)
    pollution_timestamp = datetime.fromtimestamp(pollution_raw['list'][0]['dt'])
    if is_duplicate_record(fs, "Karachi_Lyari", pollution_timestamp):
        print("Duplicate record detected. Skipping data insertion.")
        return
    time.sleep(0.1)

    weather_parsed = parse_weather_data(weather_raw)
    pollution_parsed = parse_air_pollution_data(pollution_raw)

    features_df = build_record(weather_parsed, pollution_parsed, "Karachi_Lyari")

    if features_df is not None and not features_df.empty:
        print("Collected Data:")
        print(f"Temperature: {features_df['temperature'].iloc[0]} °C")
        print(f"Humidity: {features_df['humidity'].iloc[0]}%")
        print(f"Wind Speed: {features_df['wind_speed'].iloc[0]} m/s")
        print(f"PM10 Level: {features_df['pm10'].iloc[0]} µg/m³")
        print(f"Air Quality Index: {features_df['aqi'].iloc[0]}")
        print(f"PM2.5 Level: {features_df['pm2_5'].iloc[0]} µg/m³")
        print(f"Event Timestamp: {features_df['event_timestamp'].iloc[0]}")
        print(f"AQI is Poor: {'Yes' if features_df['aqi_is_poor'].iloc[0] == 1 else 'No'}")
        print(f"PM2.5 Level Exceeds WHO: {'Yes' if features_df['pm25_exceeds_who'].iloc[0] == 1 else 'No'}")
        print("Storing in Hopsworks")
        store_features(fs, features_df)
        print("Data stored successfully")
    else:
        print("No valid data to store")

if __name__ == "__main__":
    main()
