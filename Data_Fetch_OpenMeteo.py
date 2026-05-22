import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import requests, os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import hopsworks
import pandas as pd

load_dotenv()
API_KEY_HS = os.getenv('API_KEY_HS')

def fetch_openmeteo_current(lat, lon):
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        'latitude': lat, 'longitude': lon,
        'hourly': ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
                'sulphur_dioxide', 'ozone', 'european_aqi'],
        'past_days': 0,
        'forecast_days': 1
    }
    aq_resp = requests.get(aq_url, params=aq_params).json()
    aq_hourly = aq_resp.get('hourly', {})
    
    if not aq_hourly.get('time'):
        return None

    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        'latitude': lat, 'longitude': lon,
        'hourly': ['wind_speed_10m', 'wind_direction_10m', 'temperature_2m', 'relative_humidity_2m',
                'precipitation', 'surface_pressure'],
        'past_days': 0,
        'forecast_days': 1
    }
   
    w_resp = requests.get(w_url, params=w_params).json()
    w_hourly = w_resp.get('hourly', {})

    if not w_hourly.get('time'):
        return None
    
    karachi_tz = timezone(timedelta(hours=5))
    current_hour = datetime.now(karachi_tz).hour
    
    idx = current_hour  # latest reading

    return {
        'event_timestamp': pd.to_datetime(aq_hourly['time'][idx]),
        'pm2_5': float(aq_hourly['pm2_5'][idx] or 0),
        'pm10': float(aq_hourly['pm10'][idx] or 0),
        'co': float(aq_hourly['carbon_monoxide'][idx] or 0),
        'no2': float(aq_hourly['nitrogen_dioxide'][idx] or 0),
        'so2': float(aq_hourly['sulphur_dioxide'][idx] or 0),
        'o3': float(aq_hourly['ozone'][idx] or 0),
        'european_aqi': float(aq_hourly['european_aqi'][idx] or 0),
        'wind_speed': float(w_hourly['wind_speed_10m'][idx] or 0),
        'wind_direction': float(w_hourly['wind_direction_10m'][idx] or 0),
        'temperature': float(w_hourly['temperature_2m'][idx] or 0),
        'humidity': float(w_hourly['relative_humidity_2m'][idx] or 0),
        'precipitation': float(w_hourly['precipitation'][idx] or 0),
        'pressure': float(w_hourly['surface_pressure'][idx] or 0),
        'location': 'Karachi_Lyari',
        'aqi_is_poor': int(float(aq_hourly['european_aqi'][idx] or 0) > 60),
        'pm25_exceeds_who': int(float(aq_hourly['pm2_5'][idx] or 0) > 15)
    }

def store_features(fs, record):
    if record is None:
        print("No data to store")
        return
    
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_openmeteo",
        version=1,
        description="Karachi pollution + weather from Open-Meteo",
        primary_key=['location', 'event_timestamp'],
        event_time='event_timestamp',
        online_enabled=True
    )

    try:
        df = pd.DataFrame([record])
        fg.insert(df, wait=False)
        print(f"Stored record for {record['event_timestamp']}")

    except Exception as e:
        if "already exists" in str(e):
            print(f"Duplicate record for {record['event_timestamp']}, skipping insert")
        else:
            print(f"Error occurred while storing features: {e}")

def main():
    project = hopsworks.login(
        api_key_value=API_KEY_HS,
        project="Pearls_AQI_Predictor12",
        host="eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    
    LAT, LON = 24.8607, 67.0011
    record = fetch_openmeteo_current(LAT, LON)
    
    if record:
        print(f"Temp: {record['temperature']:.1f}°C | "
              f"Wind: {record['wind_speed']:.1f} m/s | "
              f"PM2.5: {record['pm2_5']:.1f} | "
              f"AQI: {record['european_aqi']}")
        store_features(fs, record)
    else:
        print("Failed to fetch")

if __name__ == "__main__":
    main()