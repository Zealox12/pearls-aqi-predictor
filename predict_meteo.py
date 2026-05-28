# predict_meteo.py — HORIZON VERSION
import sys, types
import json
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np

load_dotenv()

print("📦 Loading models...")
project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
mr = project.get_model_registry()

# Load 3 horizon models
models = {}
for day in ['day1', 'day2', 'day3']:
    model_obj = mr.get_model(f"karachi_aqi_{day}", version=20260528)
    model_obj.download()
    models[day] = joblib.load(f'meteo/ridge_{day}.pkl')
print("✅ Models loaded")

fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_openmeteo", version=1)
df = fg.read(online=True)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
latest = df.sort_values('event_timestamp', ascending=False).iloc[0]

# Features for each day
features_day1 = [
    'pm2_5', 'pm10', 'so2', 'co', 'no2',
    'temperature', 'humidity', 'wind_speed', 'pressure',
    'pm25_3d_ago', 'pm10_3d_ago', 'pm25_7d_ago', 'pm10_7d_ago',
    'temp_3d_ago', 'wind_3d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'hour_of_day', 'month', 'is_weekend', 'is_may', 'is_clean_month'
]

features_day2 = [
    'pm25_lag_24h', 'pm10_lag_24h',
    'pm25_3d_ago', 'pm10_3d_ago', 'pm25_7d_ago', 'pm10_7d_ago',
    'temp_3d_ago', 'wind_3d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'hour_of_day', 'month', 'is_weekend', 'is_may', 'is_clean_month'
]

features_day3 = [
    'pm25_lag_48h', 'pm10_lag_48h',
    'pm25_3d_ago', 'pm10_3d_ago', 'pm25_7d_ago', 'pm10_7d_ago',
    'temp_3d_ago', 'wind_3d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'hour_of_day', 'month', 'is_weekend', 'is_may', 'is_clean_month'
]

predictions = []

# Day 1: 24 forecasts (hours 0-23)
for hour in range(24):
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)
    past_3d = df[df['event_timestamp'] <= future_time - timedelta(hours=72)].iloc[-1]
    past_7d = df[df['event_timestamp'] <= future_time - timedelta(hours=168)].iloc[-1]
    
    features = {
        'pm2_5': latest['pm2_5'], 'pm10': latest['pm10'],
        'so2': latest['so2'], 'co': latest['co'], 'no2': latest['no2'],
        'temperature': latest['temperature'], 'humidity': latest['humidity'],
        'wind_speed': latest['wind_speed'], 'pressure': latest['pressure'],
        'pm25_3d_ago': past_3d['pm2_5'], 'pm10_3d_ago': past_3d['pm10'],
        'pm25_7d_ago': past_7d['pm2_5'], 'pm10_7d_ago': past_7d['pm10'],
        'temp_3d_ago': past_3d['temperature'], 'wind_3d_ago': past_3d['wind_speed'],
        'temp_7d_ago': past_7d['temperature'], 'wind_7d_ago': past_7d['wind_speed'],
        'hour_of_day': future_time.hour, 'month': future_time.month,
        'is_weekend': 1 if future_time.weekday() >= 5 else 0,
        'is_may': 1 if future_time.month == 5 else 0,
        'is_clean_month': 1 if future_time.month == 9 else 0
    }
    
    X_pred = pd.DataFrame([features])[features_day1].fillna(0)
    aqi = models['day1'].predict(X_pred)[0]
    aqi_class = 1 if aqi <= 20 else 2 if aqi <= 40 else 3 if aqi <= 60 else 4 if aqi <= 80 else 5
    
    predictions.append({
        'event_timestamp': future_time.isoformat(), 'hour': future_time.hour,
        'day': future_time.strftime('%A'), 'date': future_time.strftime('%Y-%m-%d'),
        'european_aqi': round(float(aqi), 1), 'aqi_class': aqi_class,
        'is_poor': 1 if aqi_class >= 3 else 0
    })

# Day 2: use yesterday's actuals (from latest 24h ago)
for hour in range(24, 48):
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)
    past_24h = df[df['event_timestamp'] <= future_time - timedelta(hours=24)].iloc[-1]
    past_3d = df[df['event_timestamp'] <= future_time - timedelta(hours=72)].iloc[-1]
    past_7d = df[df['event_timestamp'] <= future_time - timedelta(hours=168)].iloc[-1]
    
    features = {
        'pm25_lag_24h': past_24h['pm2_5'], 'pm10_lag_24h': past_24h['pm10'],
        'pm25_3d_ago': past_3d['pm2_5'], 'pm10_3d_ago': past_3d['pm10'],
        'pm25_7d_ago': past_7d['pm2_5'], 'pm10_7d_ago': past_7d['pm10'],
        'temp_3d_ago': past_3d['temperature'], 'wind_3d_ago': past_3d['wind_speed'],
        'temp_7d_ago': past_7d['temperature'], 'wind_7d_ago': past_7d['wind_speed'],
        'hour_of_day': future_time.hour, 'month': future_time.month,
        'is_weekend': 1 if future_time.weekday() >= 5 else 0,
        'is_may': 1 if future_time.month == 5 else 0,
        'is_clean_month': 1 if future_time.month == 9 else 0
    }
    
    X_pred = pd.DataFrame([features])[features_day2].fillna(0)
    aqi = models['day2'].predict(X_pred)[0]
    aqi_class = 1 if aqi <= 20 else 2 if aqi <= 40 else 3 if aqi <= 60 else 4 if aqi <= 80 else 5
    
    predictions.append({
        'event_timestamp': future_time.isoformat(), 'hour': future_time.hour,
        'day': future_time.strftime('%A'), 'date': future_time.strftime('%Y-%m-%d'),
        'european_aqi': round(float(aqi), 1), 'aqi_class': aqi_class,
        'is_poor': 1 if aqi_class >= 3 else 0
    })

# Day 3: use 2-day old actuals
for hour in range(48, 72):
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)
    past_48h = df[df['event_timestamp'] <= future_time - timedelta(hours=48)].iloc[-1]
    past_3d = df[df['event_timestamp'] <= future_time - timedelta(hours=72)].iloc[-1]
    past_7d = df[df['event_timestamp'] <= future_time - timedelta(hours=168)].iloc[-1]
    
    features = {
        'pm25_lag_48h': past_48h['pm2_5'], 'pm10_lag_48h': past_48h['pm10'],
        'pm25_3d_ago': past_3d['pm2_5'], 'pm10_3d_ago': past_3d['pm10'],
        'pm25_7d_ago': past_7d['pm2_5'], 'pm10_7d_ago': past_7d['pm10'],
        'temp_3d_ago': past_3d['temperature'], 'wind_3d_ago': past_3d['wind_speed'],
        'temp_7d_ago': past_7d['temperature'], 'wind_7d_ago': past_7d['wind_speed'],
        'hour_of_day': future_time.hour, 'month': future_time.month,
        'is_weekend': 1 if future_time.weekday() >= 5 else 0,
        'is_may': 1 if future_time.month == 5 else 0,
        'is_clean_month': 1 if future_time.month == 9 else 0
    }
    
    X_pred = pd.DataFrame([features])[features_day3].fillna(0)
    aqi = models['day3'].predict(X_pred)[0]
    aqi_class = 1 if aqi <= 20 else 2 if aqi <= 40 else 3 if aqi <= 60 else 4 if aqi <= 80 else 5
    
    predictions.append({
        'event_timestamp': future_time.isoformat(), 'hour': future_time.hour,
        'day': future_time.strftime('%A'), 'date': future_time.strftime('%Y-%m-%d'),
        'european_aqi': round(float(aqi), 1), 'aqi_class': aqi_class,
        'is_poor': 1 if aqi_class >= 3 else 0
    })

# Daily summary + save (same as before)
daily_summary = {}
for day in range(3):
    day_date = (latest['event_timestamp'] + timedelta(days=day + 1)).strftime('%Y-%m-%d')
    day_preds = [p for p in predictions if p['date'] == day_date]
    if day_preds:
        poor_hours = sum(p['is_poor'] for p in day_preds)
        total_hours = len(day_preds)
        daily_summary[day_date] = {
            'date': day_date, 'day_name': day_preds[0]['day'],
            'poor_hours': f"{poor_hours}/{total_hours}",
            'poor_percentage': round((poor_hours / total_hours) * 100, 2),
            'avg_aqi': round(np.mean([p['european_aqi'] for p in day_preds]), 1),
            'max_aqi_class': max(p['aqi_class'] for p in day_preds),
            'severity': 'Hazardous' if max(p['aqi_class'] for p in day_preds) >= 5 else
                       'Very Poor' if max(p['aqi_class'] for p in day_preds) >= 4 else
                       'Poor' if max(p['aqi_class'] for p in day_preds) >= 3 else
                       'Fair' if max(p['aqi_class'] for p in day_preds) >= 2 else 'Good',
            'hourly': day_preds
        }

print("\n" + "="*55)
print("📅 3-DAY KARACHI AQI FORECAST (Horizon Models)")
print("="*55)
for date, summary in daily_summary.items():
    print(f"\n{summary['day_name']}, {date}")
    print(f"  Alert:      {summary['severity']}")
    print(f"  Poor Hours: {summary['poor_hours']} ({summary['poor_percentage']}%)")
    print(f"  Avg AQI:    {summary['avg_aqi']}")
    print(f"  Hourly:")
    for p in summary['hourly'][::3]:
        status = "🔴" if p['is_poor'] else "🟢"
        print(f"    {p['hour']:02d}:00 {status} AQI={p['european_aqi']:.1f} (Class {p['aqi_class']})")

forecast_output = {
    'generated_at': datetime.now().isoformat(),
    'daily': {k: {kk: vv for kk, vv in v.items() if kk != 'hourly'} for k, v in daily_summary.items()},
    'hourly': predictions
}
with open('meteo/aqi_forecast.json', 'w') as f:
    json.dump(forecast_output, f, indent=2)
print(f"\n✅ Forecast saved to meteo/aqi_forecast.json")