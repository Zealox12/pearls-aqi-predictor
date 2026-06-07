# train_openmeteo.py — RECURSIVE MULTI-OUTPUT VERSION
import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, json, joblib
from datetime import datetime
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

load_dotenv()

print("Loading data from Hopsworks...")
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
df = df.sort_values('event_timestamp').reset_index(drop=True)
df['year'] = df['event_timestamp'].dt.year
df['month'] = df['event_timestamp'].dt.month
df['hour_of_day'] = df['event_timestamp'].dt.hour
df = df[df['year'] >= 2023]

# Lag features
df['pm25_3d_ago'] = df['pm2_5'].shift(72)
df['pm10_3d_ago'] = df['pm10'].shift(72)
df['temp_3d_ago'] = df['temperature'].shift(72)
df['wind_3d_ago'] = df['wind_speed'].shift(72)
df['pm25_7d_ago'] = df['pm2_5'].shift(168)
df['pm10_7d_ago'] = df['pm10'].shift(168)
df['temp_7d_ago'] = df['temperature'].shift(168)
df['wind_7d_ago'] = df['wind_speed'].shift(168)

df['is_clean_month'] = (df['month'] == 9).astype(int)
df['is_weekend'] = df['event_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_may'] = (df['month'] == 5).astype(int)

# Targets
df['aqi_tomorrow'] = df['european_aqi'].shift(-24)
df['pm25_tomorrow'] = df['pm2_5'].shift(-24)
df['pm10_tomorrow'] = df['pm10'].shift(-24)
df['so2_tomorrow'] = df['so2'].shift(-24)
df['co_tomorrow'] = df['co'].shift(-24)
df['no2_tomorrow'] = df['no2'].shift(-24)

target_cols = ['aqi_tomorrow', 'pm25_tomorrow', 'pm10_tomorrow', 
               'so2_tomorrow', 'co_tomorrow', 'no2_tomorrow']

features = [
    'pm2_5', 'pm10', 'so2', 'co', 'no2',
    'temperature', 'humidity', 'wind_speed', 'pressure',
    'pm25_3d_ago', 'pm10_3d_ago', 'pm25_7d_ago', 'pm10_7d_ago',
    'temp_3d_ago', 'wind_3d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'hour_of_day', 'month', 'is_weekend', 'is_may', 'is_clean_month'
]

# Train/test split
df_model = df.dropna(subset=features + target_cols)
train = df_model[df_model['year'].isin([2023, 2024])]
test = df_model[df_model['year'] >= 2025]

models = {
    'Ridge': MultiOutputRegressor(Ridge(alpha=1.0)),
    'Lasso': MultiOutputRegressor(Lasso(alpha=0.1, max_iter=5000)),
    'RandomForest': MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)),
    'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)),
    'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=50, max_depth=3, random_state=42)),
    'LightGBM': MultiOutputRegressor(LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)),
}

results = {}
best_model = None
best_mae = float('inf')
best_name = ""

print("\nTraining models...")

for name, model in models.items():
    model.fit(train[features].fillna(0), train[target_cols])
    preds = model.predict(test[features].fillna(0))
    mae = mean_absolute_error(test['aqi_tomorrow'], preds[:, 0])
    results[name] = round(mae, 4)
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name
    
    print(f"  {name:20s} | MAE: {mae:.4f}")

persist = mean_absolute_error(test['aqi_tomorrow'], test['european_aqi'])
print(f"  {'Persistence':20s} | MAE: {persist:.4f}")
print(f"\nBest: {best_name} (MAE: {best_mae:.4f}, {((persist-best_mae)/persist*100):.1f}% improvement)")

import os as _os
_os.makedirs('meteo', exist_ok=True)

for name, m in models.items():
    joblib.dump(m, f'meteo/{name}.pkl')

joblib.dump(best_model, 'meteo/recursive_model.pkl')
joblib.dump(features, 'meteo/features.pkl')
joblib.dump(target_cols, 'meteo/target_cols.pkl')
print("All models saved to meteo/")

mr = project.get_model_registry()
version = int(datetime.now().strftime('%Y%m%d'))

for name, m in models.items():
    try:
        old = mr.get_model(f"karachi_aqi_{name}", version=version)
        old.delete()
    except: pass
    reg_obj = mr.sklearn.create_model(
        name=f"karachi_aqi_{name}",
        version=version,
        metrics={'MAE': results[name]}
    )
    reg_obj.save(f'meteo/{name}.pkl')
    print(f" karachi_aqi_{name} v{version} (MAE: {results[name]})")

# Register best as production
try:
    old = mr.get_model("karachi_aqi_recursive", version=version)
    old.delete()
except: pass

reg_obj = mr.sklearn.create_model(
    name="karachi_aqi_recursive",
    version=version,
    description=f"Best model: {best_name}, MAE: {best_mae:.4f}",
    metrics={'MAE': results[best_name]}
)
reg_obj.save('meteo/recursive_model.pkl')
print(f"  karachi_aqi_recursive v{version} ({best_name}, MAE: {best_mae:.4f})")

comparison = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_name,
    'best_mae': best_mae,
    'persistence_mae': persist,
    'improvement_pct': round((persist - best_mae) / persist * 100, 1),
    'features': features,
    'results': results
}
with open('meteo/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"\nTraining complete! Best: {best_name} (MAE: {best_mae:.4f})")