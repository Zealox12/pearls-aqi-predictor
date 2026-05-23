import sys, types
import json

m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, joblib, requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np

load_dotenv()

print("Loading model from Hopsworks...")
project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
mr = project.get_model_registry()

model_obj = mr.get_model("karachi_aqi_om_production", version=20260523)
model_obj.download()
model = joblib.load('meteo/best_model.pkl')
print("Model loaded")

fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_openmeteo", version=1)
df = fg.read(online=True)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
latest = df.sort_values('event_timestamp', ascending=False).iloc[0]

print(f"Latest data: {latest['event_timestamp']}")

lat, lon = 24.8607, 67.0011

final_features = [
    'pm25_3d_ago', 'pm10_3d_ago', 'temp_3d_ago', 'wind_3d_ago',
    'pm25_7d_ago', 'pm10_7d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'month', 'hour_of_day',
    'is_clean_month', 'is_weekend', 'is_may'
]

predictions = []

for hour in range(72):
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)

    target_3d = future_time - timedelta(hours=72)
    target_7d = future_time - timedelta(hours=168)
    past_3d = df[df['event_timestamp'] <= target_3d].iloc[-1]
    past_7d = df[df['event_timestamp'] <= target_7d].iloc[-1]   
        
    features = {
        'pm25_3d_ago': past_3d['pm2_5'],
        'pm10_3d_ago': past_3d['pm10'],
        'temp_3d_ago': past_3d['temperature'],
        'wind_3d_ago': past_3d['wind_speed'],
        'pm25_7d_ago': past_7d['pm2_5'],
        'pm10_7d_ago': past_7d['pm10'],
        'temp_7d_ago': past_7d['temperature'],
        'wind_7d_ago': past_7d['wind_speed'],
        'month': future_time.month,
        'hour_of_day': future_time.hour,
        'is_clean_month': 1 if future_time.month == 9 else 0,
        'is_weekend': 1 if future_time.weekday() >= 5 else 0,
        'is_may': 1 if future_time.month == 5 else 0
    }
    
    X_pred = pd.DataFrame([features])[final_features].fillna(0)
    european_aqi = model.predict(X_pred)[0]
    
    if european_aqi <= 20: aqi_class = 1
    elif european_aqi <= 40: aqi_class = 2
    elif european_aqi <= 60: aqi_class = 3
    elif european_aqi <= 80: aqi_class = 4
    else: aqi_class = 5

    print(f"Hour {hour}: 3d_ago_pm25={past_3d['pm2_5']:.1f}, 7d_ago_pm25={past_7d['pm2_5']:.1f}, forecast_pm25={european_aqi:.1f}")
    
    predictions.append({
        'event_timestamp': future_time.isoformat(),
        'hour': future_time.hour,
        'day': future_time.strftime('%A'),
        'date': future_time.strftime('%Y-%m-%d'),
        'european_aqi': round(float(european_aqi), 1),
        'aqi_class': aqi_class,
        'is_poor': 1 if aqi_class >= 3 else 0
    })

# Print first 3 days of lag values to verify variation
print("\n=== Lag Verification ===")
for hour in [0, 24, 48]:
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)
    target_3d = future_time - timedelta(hours=72)
    target_7d = future_time - timedelta(hours=168)
    past_3d = df[df['event_timestamp'] <= target_3d].iloc[-1]
    past_7d = df[df['event_timestamp'] <= target_7d].iloc[-1]
    print(f"Predicting {future_time.date()}:")
    print(f"  3d_ago ({target_3d.date()}): PM2.5={past_3d['pm2_5']:.1f}")
    print(f"  7d_ago ({target_7d.date()}): PM2.5={past_7d['pm2_5']:.1f}")


daily_summary = {}
for day in range(3):
    day_date = (latest['event_timestamp'] + timedelta(days=day + 1)).strftime('%Y-%m-%d')
    day_preds = [p for p in predictions if p['date'] == day_date]
    
    if day_preds:
        poor_hours = sum(p['is_poor'] for p in day_preds)
        total_hours = len(day_preds)
        avg_aqi = np.mean([p['european_aqi'] for p in day_preds])
        max_aqi = max(p['aqi_class'] for p in day_preds)
        
        if max_aqi >= 5: severity = 'Hazardous'
        elif max_aqi >= 4: severity = 'Very Poor'
        elif max_aqi >= 3: severity = 'Poor'
        elif max_aqi >= 2: severity = 'Fair'
        else: severity = 'Good'
        
        daily_summary[day_date] = {
            'date': day_date,
            'day_name': day_preds[0]['day'],
            'poor_hours': f"{poor_hours}/{total_hours}",
            'poor_percentage': round((poor_hours / total_hours) * 100, 2),
            'avg_aqi': round(avg_aqi, 1),
            'max_aqi_class': max_aqi,
            'severity': severity,
            'hourly': day_preds
        }

for date, summary in daily_summary.items():
    print(f"\n{summary['day_name']}, {date}")
    print(f"  Alert:      {summary['severity']}")
    print(f"  Poor Hours: {summary['poor_hours']} ({summary['poor_percentage']}%)")
    print(f"  Avg AQI:    {summary['avg_aqi']}")
    print(f"  Max AQI:    {summary['max_aqi_class']}")
    print(f"  Hourly:")
    for p in summary['hourly'][::3]:
        status = "Y" if p['is_poor'] else "N"
        print(f"    {p['hour']:02d}:00 {status} AQI={p['european_aqi']:.1f} (Class {p['aqi_class']})")

forecast_output = {
    'generated_at': datetime.now().isoformat(),
    'daily': {k: {kk: vv for kk, vv in v.items() if kk != 'hourly'} 
              for k, v in daily_summary.items()},
    'hourly': predictions
}

with open('meteo/aqi_forecast.json', 'w') as f:
    json.dump(forecast_output, f, indent=2)

print(f"\nForecast saved to meteo/aqi_forecast.json")

# Quick validation
import requests

lat, lon = 24.8607, 67.0011
resp = requests.get(
    "https://air-quality-api.open-meteo.com/v1/air-quality",
    params={
        'latitude': lat, 'longitude': lon,
        'hourly': ['european_aqi', 'pm2_5', 'pm10'],
        'forecast_days': 3,
        'timezone': 'Asia/Karachi'
    }
).json()

# Print first few forecast hours
hourly = resp['hourly']
print("Open-Meteo's own forecast:")
for i in range(0, 24, 3):
    print(f"  {hourly['time'][i]}: AQI={hourly['european_aqi'][i]}, PM2.5={hourly['pm2_5'][i]}, PM10={hourly['pm10'][i]}")

# Compare with your predictions
print("\nYour predictions:")
for i in range(0, 24, 3):
    print(f"  Hour {i}: AQI={predictions[i]['european_aqi']}")