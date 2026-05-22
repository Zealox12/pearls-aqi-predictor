# backfill_openmeteo.py
import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import requests, os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import hopsworks
import pandas as pd
import time

load_dotenv()
API_KEY_HS = os.getenv('API_KEY_HS')

def fetch_openmeteo_chunk(start_date, end_date, lat, lon):
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 
                   'sulphur_dioxide', 'ozone', 'european_aqi'],
        'timezone': 'Asia/Karachi'
    }
    aq_resp = requests.get(aq_url, params=aq_params).json()
    aq_hourly = aq_resp.get('hourly', {})
    
    if not aq_hourly.get('time'):
        return None

    w_url = "https://archive-api.open-meteo.com/v1/archive"
    w_params = {
        'latitude': lat, 'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ['wind_speed_10m', 'wind_direction_10m', 'temperature_2m',
                   'relative_humidity_2m', 'precipitation', 'surface_pressure'],
        'timezone': 'Asia/Karachi'
    }
    w_resp = requests.get(w_url, params=w_params).json()
    w_hourly = w_resp.get('hourly', {})
    
    if not w_hourly.get('time'):
        return None
    
    if len(aq_hourly['time']) == 0 or len(w_hourly['time']) == 0:
        return None

    # Build DataFrame
    df = pd.DataFrame({
        'event_timestamp': pd.to_datetime(aq_hourly['time']),
        'pm2_5': pd.Series(pd.to_numeric(aq_hourly['pm2_5'], errors='coerce')).fillna(0).astype(float),
        'pm10': pd.Series(pd.to_numeric(aq_hourly['pm10'], errors='coerce')).fillna(0).astype(float),
        'co': pd.Series(pd.to_numeric(aq_hourly['carbon_monoxide'], errors='coerce')).fillna(0).astype(float),
        'no2': pd.Series(pd.to_numeric(aq_hourly['nitrogen_dioxide'], errors='coerce')).fillna(0).astype(float),
        'so2': pd.Series(pd.to_numeric(aq_hourly['sulphur_dioxide'], errors='coerce')).fillna(0).astype(float),
        'o3': pd.Series(pd.to_numeric(aq_hourly['ozone'], errors='coerce')).fillna(0).astype(float),
        'european_aqi': pd.Series(pd.to_numeric(aq_hourly['european_aqi'], errors='coerce')).fillna(0).astype(float),
        'wind_speed': pd.Series(pd.to_numeric(w_hourly['wind_speed_10m'], errors='coerce')).fillna(0).astype(float),
        'wind_direction': pd.Series(pd.to_numeric(w_hourly['wind_direction_10m'], errors='coerce')).fillna(0).astype(float),
        'temperature': pd.Series(pd.to_numeric(w_hourly['temperature_2m'], errors='coerce')).fillna(0).astype(float),
        'humidity': pd.Series(pd.to_numeric(w_hourly['relative_humidity_2m'], errors='coerce')).fillna(0).astype(float),
        'precipitation': pd.Series(pd.to_numeric(w_hourly['precipitation'], errors='coerce')).fillna(0).astype(float),
        'pressure': pd.Series(pd.to_numeric(w_hourly['surface_pressure'], errors='coerce')).fillna(0).astype(float),
    })
    
    df['location'] = 'Karachi_Lyari'
    df['aqi_is_poor'] = (df['european_aqi'] > 60).astype(int).astype('int64')
    df['pm25_exceeds_who'] = (df['pm2_5'] > 15).astype(int).astype('int64')
    
    return df

def store_features(fs, df):
    if df is None or df.empty:
        return
    
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_openmeteo",
        version=1,
        description="Karachi pollution + weather from Open-Meteo",
        primary_key=['location', 'event_timestamp'],
        event_time='event_timestamp',
        online_enabled=True
    )
    
    fg.insert(df, wait=False)
    print(f" Stored {len(df)} records")

def main():
    project = hopsworks.login(
        api_key_value=API_KEY_HS,
        project="Pearls_AQI_Predictor12",
        host="eu-west.cloud.hopsworks.ai"
    )
    fs = project.get_feature_store()
    print("Connected to Hopsworks\n")
    
    LAT, LON = 24.8607, 67.0011
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    for year in range(2013, 2027):
        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < 2026 else end_date
        
        print(f"{start} → {end}...", end=" ")
        
        df = fetch_openmeteo_chunk(start, end, LAT, LON)
        if df is not None and not df.empty:
            store_features(fs, df)
        else:
            print("No data")
        
        time.sleep(1)
    
    print("\nBackfill complete!")

if __name__ == "__main__":
    main()