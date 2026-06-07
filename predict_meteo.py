import subprocess
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

project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
mr = project.get_model_registry()
model_dir = mr.get_model("karachi_aqi_recursive").download()
model = joblib.load(os.path.join(model_dir, 'recursive_model.pkl'))
features = joblib.load('meteo/features.pkl')
print("Model loaded")

fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_openmeteo", version=1)
df = fg.read(online=True)
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df = df.sort_values('event_timestamp').reset_index(drop=True)
latest = df.sort_values('event_timestamp', ascending=False).iloc[0]

def get_features_for_timestamp(df, target_time):
    slice_3d = df[df['event_timestamp'] <= target_time - timedelta(hours=72)]
    slice_7d = df[df['event_timestamp'] <= target_time - timedelta(hours=168)]
    past_3d = slice_3d.iloc[-1] if len(slice_3d) > 0 else df.iloc[0]  # ← fallback
    past_7d = slice_7d.iloc[-1] if len(slice_7d) > 0 else df.iloc[0]  # ← fallback
    
    return {
        'pm2_5': latest['pm2_5'], 'pm10': latest['pm10'],
        'so2': latest['so2'], 'co': latest['co'], 'no2': latest['no2'],
        'temperature': latest['temperature'], 'humidity': latest['humidity'],
        'wind_speed': latest['wind_speed'], 'pressure': latest['pressure'],
        'pm25_3d_ago': past_3d['pm2_5'], 'pm10_3d_ago': past_3d['pm10'],
        'pm25_7d_ago': past_7d['pm2_5'], 'pm10_7d_ago': past_7d['pm10'],
        'temp_3d_ago': past_3d['temperature'], 'wind_3d_ago': past_3d['wind_speed'],
        'temp_7d_ago': past_7d['temperature'], 'wind_7d_ago': past_7d['wind_speed'],
        'hour_of_day': target_time.hour, 'month': target_time.month,
        'is_weekend': 1 if target_time.weekday() >= 5 else 0,
        'is_may': 1 if target_time.month == 5 else 0,
        'is_clean_month': 1 if target_time.month == 9 else 0
    }

def predict_24h(df, start_time, current_features):
    preds = []
    feat = current_features.copy()
    
    for h in range(24):
        ft = start_time + timedelta(hours=h+1)
        feat['hour_of_day'] = ft.hour
        feat['month'] = ft.month
        feat['is_weekend'] = 1 if ft.weekday() >= 5 else 0
        feat['is_may'] = 1 if ft.month == 5 else 0
        feat['is_clean_month'] = 1 if ft.month == 9 else 0
        
        X = pd.DataFrame([feat])[features].fillna(0)
        out = model.predict(X)[0]
        preds.append({'time': ft, 'aqi': out[0], 'pm2_5': out[1], 'pm10': out[2],
                      'so2': out[3], 'co': out[4], 'no2': out[5]})
    
    return preds

base_features = get_features_for_timestamp(df, latest['event_timestamp'])

# Day 1: Use today's actuals
day1_preds = predict_24h(df, latest['event_timestamp'], base_features)

# Day 2: Use Day 1 predictions
day2_features = base_features.copy()
day2_features['pm2_5'] = day1_preds[-1]['pm2_5']
day2_features['pm10'] = day1_preds[-1]['pm10']
day2_features['so2'] = day1_preds[-1]['so2']
day2_features['co'] = day1_preds[-1]['co']
day2_features['no2'] = day1_preds[-1]['no2']
day2_preds = predict_24h(df, day1_preds[-1]['time'], day2_features)

# Day 3: Use Day 2 predictions
day3_features = day2_features.copy()
day3_features['pm2_5'] = day2_preds[-1]['pm2_5']
day3_features['pm10'] = day2_preds[-1]['pm10']
day3_features['so2'] = day2_preds[-1]['so2']
day3_features['co'] = day2_preds[-1]['co']
day3_features['no2'] = day2_preds[-1]['no2']
day3_preds = predict_24h(df, day2_preds[-1]['time'], day3_features)

# Combine all predictions
all_preds = day1_preds + day2_preds + day3_preds


# Print summary
print("3-DAY KARACHI AQI FORECAST (Recursive)")
for day_offset, day_preds in enumerate([day1_preds, day2_preds, day3_preds]):
    day_name = (latest['event_timestamp'] + timedelta(days=day_offset+1)).strftime('%A')
    day_date = (latest['event_timestamp'] + timedelta(days=day_offset+1)).strftime('%Y-%m-%d')
    aqi_vals = [p['aqi'] for p in day_preds]
    print(f"\n{day_name}, {day_date}")
    print(f"  AQI Range: {min(aqi_vals):.1f} - {max(aqi_vals):.1f}")
    print(f"  Avg AQI:   {np.mean(aqi_vals):.1f}")
    for p in day_preds[::3]:
        aqi_class = 1 if p['aqi'] <= 20 else 2 if p['aqi'] <= 40 else 3 if p['aqi'] <= 60 else 4 if p['aqi'] <= 80 else 5
        status = "Red" if aqi_class >= 3 else "Green"
        print(f"    {p['time'].strftime('%H:%M')} {status} AQI={p['aqi']:.1f} (Class {aqi_class})")

forecast_output = {
    'generated_at': datetime.now().isoformat(),
    'daily': {},
    'hourly': [{'time': p['time'].isoformat(), 'aqi': round(p['aqi'], 1)} for p in all_preds]
}

from collections import defaultdict

daily_grouped = defaultdict(list)
for p in all_preds:
    date_key = p['time'].strftime('%Y-%m-%d')
    daily_grouped[date_key].append(p['aqi'])

daily_summary = {}
for date_key, aqi_vals in daily_grouped.items():
    max_aqi = max(aqi_vals)
    daily_summary[date_key] = {
        'avg_aqi': round(np.mean(aqi_vals), 1),
        'max_aqi': round(max_aqi, 1),
        'min_aqi': round(min(aqi_vals), 1),
        'aqi_class': 1 if max_aqi <= 20 else 2 if max_aqi <= 40 else 3 if max_aqi <= 60 else 4 if max_aqi <= 80 else 5
    }

forecast_output['daily'] = daily_summary
with open('meteo/aqi_forecast.json', 'w') as f:
    json.dump(forecast_output, f, indent=2)
print(f"\nForecast saved to meteo/aqi_forecast.json")

# Only push to GitHub if running in GitHub Actions
if os.environ.get('GITHUB_ACTIONS'):
    print("Pushing forecast to GitHub...")
    
    # Configure git
    subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions'])
    subprocess.run(['git', 'config', '--global', 'user.email', 'actions@github.com'])
    
    # Add, commit, and push
    subprocess.run(['git', 'add', 'meteo/aqi_forecast.json'])
    subprocess.run('git commit -m "Update AQI forecast [skip ci]" || true', shell=True)
    subprocess.run(['git', 'push'])
    
    print("Forecast pushed to GitHub!")
else:
    print("Not in GitHub Actions, skipping git push")