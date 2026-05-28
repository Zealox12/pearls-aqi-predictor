# train_openmeteo.py — HORIZON-SPECIFIC VERSION
import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, json, joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

load_dotenv()

print("📦 Loading data from Hopsworks...")
project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("karachi_aqi_openmeteo", version=1)
df = fg.read(online=True)
print(f"   {len(df)} rows loaded")

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df['year'] = df['event_timestamp'].dt.year
df['month'] = df['event_timestamp'].dt.month
df['hour_of_day'] = df['event_timestamp'].dt.hour
df = df[df['year'] >= 2023]

# ============================================
# CREATE LAG FEATURES
# ============================================
df['pm25_lag_24h'] = df['pm2_5'].shift(24)
df['pm10_lag_24h'] = df['pm10'].shift(24)
df['pm25_lag_48h'] = df['pm2_5'].shift(48)
df['pm10_lag_48h'] = df['pm10'].shift(48)
df['pm25_3d_ago'] = df['pm2_5'].shift(72)
df['pm10_3d_ago'] = df['pm10'].shift(72)
df['pm25_7d_ago'] = df['pm2_5'].shift(168)
df['pm10_7d_ago'] = df['pm10'].shift(168)
df['temp_3d_ago'] = df['temperature'].shift(72)
df['wind_3d_ago'] = df['wind_speed'].shift(72)
df['temp_7d_ago'] = df['temperature'].shift(168)
df['wind_7d_ago'] = df['wind_speed'].shift(168)

df['is_clean_month'] = (df['month'] == 9).astype(int)
df['is_weekend'] = df['event_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_may'] = (df['month'] == 5).astype(int)

# ============================================
# TARGETS FOR 3 HORIZONS
# ============================================
df['aqi_day1'] = df['european_aqi'].shift(-24)  # 24h ahead
df['aqi_day2'] = df['european_aqi'].shift(-48)  # 48h ahead
df['aqi_day3'] = df['european_aqi'].shift(-72)  # 72h ahead

# ============================================
# FEATURES FOR EACH HORIZON
# ============================================
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

# ============================================
# TRAIN 3 HORIZON-SPECIFIC MODELS
# ============================================
print("\n🔄 Training horizon-specific models...")
print("="*55)

models = {}
results = {}

for day, features, target in [
    ('Day1', features_day1, 'aqi_day1'),
    ('Day2', features_day2, 'aqi_day2'),
    ('Day3', features_day3, 'aqi_day3')
]:
    df_model = df.dropna(subset=features + [target])
    train = df_model[df_model['year'].isin([2023, 2024])]
    test = df_model[df_model['year'] >= 2025]
    
    model = Ridge(alpha=1.0)
    model.fit(train[features].fillna(0), train[target])
    preds = model.predict(test[features].fillna(0))
    mae = mean_absolute_error(test[target], preds)
    
    models[day] = model
    results[day] = {'MAE': round(mae, 2), 'features': features}
    
    persist = mean_absolute_error(test[target], test['european_aqi'])
    print(f"  {day}: MAE={mae:.2f} | Persistence={persist:.2f} | Improvement={((persist-mae)/persist*100):.1f}%")

# ============================================
# SAVE LOCALLY
# ============================================
import os as _os
_os.makedirs('meteo', exist_ok=True)

for day, model in models.items():
    joblib.dump(model, f'meteo/ridge_{day.lower()}.pkl')
    print(f"  💾 Saved: meteo/ridge_{day.lower()}.pkl")

# ============================================
# REGISTER IN HOPSWORKS
# ============================================
print("\n📦 Registering models in Hopsworks...")
mr = project.get_model_registry()
version = int(datetime.now().strftime('%Y%m%d'))

for day, model in models.items():
    joblib.dump(model, f'meteo/ridge_{day.lower()}.pkl')
    try:
        old = mr.get_model(f"karachi_aqi_{day.lower()}", version=version)
        old.delete()
    except:
        pass
    reg_obj = mr.sklearn.create_model(
        name=f"karachi_aqi_{day.lower()}",
        version=version,
        metrics={'MAE': results[day]['MAE']}
    )
    reg_obj.save(f'meteo/ridge_{day.lower()}.pkl')
    print(f"  📦 karachi_aqi_{day.lower()} v{version} (MAE: {results[day]['MAE']})")

print(f"\n✅ Training complete! 3 models registered.")