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

def get_latest_model(mr, model_name):
    for days_back in range(7):
        try_date = int((datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d"))
        try:
            return mr.get_model(model_name, version=try_date)
        except:
            continue
    raise Exception(f"Model {model_name} not found in last 7 days")

print("Loading production models from hopsworks:")
project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
mr = project.get_model_registry()

binary_model_obj = get_latest_model(mr, "karachi_aqi_production")
binary_model_obj.download()
binary_model = joblib.load('best_model.pkl')

multi_model_obj = get_latest_model(mr, "karachi_aqi_multiclass_production")
multi_model_obj.download()
multi_model = joblib.load('best_multi_model.pkl')

print("Models loaded successfully.")

fs = project.get_feature_store()
fg = fs.get_feature_group("weather_pollution_features", version=1)
df = fg.read()
print(f"Fetched {len(df)} records from feature store")

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
latest = df.sort_values('event_timestamp', ascending=False).iloc[0]

import requests
API_KEY = os.getenv('API_KEY')
lat, lon = 24.8607, 67.0011

forecast_response = requests.get(
    "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
    params={'lat': lat, 'lon': lon, 'appid': API_KEY}
)

forecast = forecast_response.json().get('list', []) if forecast_response.status_code == 200 else []
print(f"Fetched {len(forecast)} forecast records from OpenWeather API")

feature_cols = ['pm2_5', 'pm10', 'so2', 'co', 'no2',
                'hour_of_day', 'day_of_week', 'month', 'year',
                'is_winter', 'is_dust_event', 'is_clean_month']

predictions = []
validation = []

for hour in range(72): #3 days
    future_time = latest['event_timestamp'] + timedelta(hours=hour + 1)

    if hour < len(forecast):
        f = forecast[hour]
        pm25 = f['components']['pm2_5']
        pm10 = f['components']['pm10']
        co = f['components']['co']
        no2 = f['components']['no2']
        o3 = f['components']['o3']
        so2 = f['components']['so2']
    else:
        pm25 = latest['pm2_5']
        pm10 = latest['pm10']
        co = latest['co']
        no2 = latest['no2']
        o3 = latest['o3']
        so2 = latest['so2']
    features = {
        'pm2_5': pm25,
        'pm10': pm10,
        'so2': so2,
        'co': co,
        'no2': no2,
        'hour_of_day': future_time.hour,
        'day_of_week': future_time.dayofweek,
        'month': future_time.month,
        'year': future_time.year,
        'is_winter': 1 if future_time.month in [11, 12, 1, 2] else 0,
        'is_dust_event': 1 if (latest['pm10'] > 100 and latest['pm2_5'] < 50) else 0,
        'is_clean_month': 1 if future_time.month == 9 else 0
    }

    X_pred = pd.DataFrame([features])[feature_cols].fillna(0)

    is_poor = binary_model.predict(X_pred)[0]

    aqi_class = multi_model.predict(X_pred)[0] + 1

    openw_aqi = None
    if hour < len(forecast):
        openw_aqi = forecast[hour].get('main', {}).get('aqi')

    predictions.append({
        'event_timestamp': future_time.isoformat(),
        'hour': future_time.hour,
        'day' : future_time.strftime('%A'),
        'date': future_time.strftime('%Y-%m-%d'),
        'is_poor': int(is_poor),
        'aqi_class': int(aqi_class),
        'openw_aqi': openw_aqi
    })

daily_summary = {}

for day in range(3):
    day_date = (latest['event_timestamp'] + timedelta(days=day + 1)).strftime('%Y-%m-%d')
    day_preds = [p for p in predictions if p['date'] == day_date]

    if day_preds:
        poor_hours = sum(p['is_poor'] for p in day_preds)
        total_hours = len(day_preds)
        avg_aqi_class = np.mean([p['aqi_class'] for p in day_preds])
        max_aqi_class = np.max([p['aqi_class'] for p in day_preds])

        if max_aqi_class >= 5:
            severity = 'Hazardous'
        elif max_aqi_class >= 4:
            severity = 'Very Poor'
        elif max_aqi_class >= 3:
            severity = 'Poor'
        elif max_aqi_class >= 2:
            severity = 'Fair'
        else:
            severity = 'Good'

        openw_poor = sum(1 for p in day_preds if p['openw_aqi'] is not None and p['openw_aqi'] >= 3)
        openw_total = sum(1 for p in day_preds if p['openw_aqi'] is not None)

        daily_summary[day_date] = {
            'date': day_date,
            'day_name': day_preds[0]['day'],
            'poor_hours': f"{poor_hours}/{total_hours}",
            'poor_percentage': round((poor_hours / total_hours) * 100, 2),
            'avg_aqi_class': round(avg_aqi_class, 1),
            'max_aqi_class': int(max_aqi_class),
            'severity': severity,
            'openw_poor_validation': f"{openw_poor}/{openw_total}" if openw_total > 0 else "N/A",
            'hourly': day_preds
        }

print("\n3-Day AQI karachi Forecast Summary:")

for date, summary in daily_summary.items():
    print(f"\n{summary['day_name']}, {date}")
    print(f"  Alert:      {summary['severity']}")
    print(f"  Poor Hours: {summary['poor_hours']} ({summary['poor_percentage']}%)")
    print(f"  Avg AQI:    {summary['avg_aqi_class']}")
    print(f"  Max AQI:    {summary['max_aqi_class']}")
    print(f"  OW Forecast:{summary['openw_poor_validation']}")

    print(f"  Hourly:")
    for p in summary['hourly'][::3]:  # Every 3 hours
        ow = f" (OW:{p['openw_aqi']})" if p['openw_aqi'] else ""
        status = "🔴" if p['is_poor'] else "🟢"
        print(f"    {p['hour']:02d}:00 {status} AQI={p['aqi_class']}{ow}")

forecast_output = {
    'generated_at': datetime.now().isoformat(),
    'daily': {k: {kk: vv for kk, vv in v.items() if kk != 'hourly'} 
              for k, v in daily_summary.items()},
    'hourly': predictions
}

with open('aqi_forecast.json', 'w') as f:
    json.dump(forecast_output, f, indent=2)

print("\nForecast saved to aqi_forecast.json")

if forecast:
    matched = [p for p in predictions if p['openw_aqi'] is not None and p['aqi_class'] == p['openw_aqi']]
    if matched:
        correct = sum(1 for p in matched if p['is_poor'] == (p['openw_aqi'] >= 3))
        print(f"\nValidation against OpenWeather API: {correct}/{len(matched)} hourly predictions matched in AQI class and poor/good status.")

print("\nPrediction process completed.")