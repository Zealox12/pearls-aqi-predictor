import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, json, joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
print(f" {len(df)} rows loaded")

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df['year'] = df['event_timestamp'].dt.year
df['month'] = df['event_timestamp'].dt.month
df['hour_of_day'] = df['event_timestamp'].dt.hour

df = df[df['year'] >= 2023]

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

df['aqi_tomorrow'] = df['european_aqi'].shift(-24)

final_features = [
    'pm25_3d_ago', 'pm10_3d_ago', 'temp_3d_ago', 'wind_3d_ago',
    'pm25_7d_ago', 'pm10_7d_ago', 'temp_7d_ago', 'wind_7d_ago',
    'month', 'hour_of_day',
    'is_clean_month', 'is_weekend', 'is_may'
]

df_model = df.dropna(subset=final_features + ['aqi_tomorrow'])
train = df_model[df_model['year'].isin([2023, 2024])]
test = df_model[df_model['year'] >= 2025]

X_train = train[final_features].fillna(0)
y_train = train['aqi_tomorrow']
X_test = test[final_features].fillna(0)
y_test = test['aqi_tomorrow']

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

print("\nTraining models")

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=50, max_depth=3, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1),
}

results = {}
best_model = None
best_mae = float('inf')
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results[name] = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    }
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name
    
    print(f"  {name:20s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

persist_mae = mean_absolute_error(y_test, test['european_aqi'])
print(f"  {'Persistence':20s} | MAE: {persist_mae:.4f}")
print(f"\nBest: {best_name} (MAE: {best_mae:.4f}, {((persist_mae-best_mae)/persist_mae*100):.1f}% better than persistence)")

import os as _os
_os.makedirs('meteo', exist_ok=True)

for name, model_obj in models.items():
    joblib.dump(model_obj, f'meteo/{name}.pkl')
    print(f" Saved: meteo/{name}.pkl")

joblib.dump(best_model, 'meteo/best_model.pkl')

comparison = {
    'timestamp': datetime.now().isoformat(),
    'best_model': best_name,
    'best_mae': best_mae,
    'persistence_mae': persist_mae,
    'improvement_pct': round((persist_mae - best_mae) / persist_mae * 100, 1),
    'features': final_features,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'results': {k: v for k, v in results.items()}
}
with open('meteo/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print(" Saved: meteo/model_comparison.json")

print("\nRegistering models in Hopsworks...")
mr = project.get_model_registry()
version = int(datetime.now().strftime('%Y%m%d'))

for name, model_obj in models.items():
    joblib.dump(model_obj, f'meteo/{name}.pkl')
    try:
        old = mr.get_model(f"karachi_aqi_om_{name}", version=version)
        old.delete()
    except:
        pass
    reg_obj = mr.sklearn.create_model(
        name=f"karachi_aqi_om_{name}",
        version=version,
        metrics=results[name]
    )
    reg_obj.save(f'meteo/{name}.pkl')
    print(f" karachi_aqi_om_{name} v{version}")

joblib.dump(best_model, 'meteo/best_model.pkl')
try:
    old = mr.get_model("karachi_aqi_om_production", version=version)
    old.delete()
except:
    pass
best_obj = mr.sklearn.create_model(
    name="karachi_aqi_om_production",
    version=version,
    description=f"Production: {best_name}, MAE: {best_mae:.4f}",
    metrics={'MAE': best_mae, 'RMSE': results[best_name]['RMSE'], 'R2': results[best_name]['R2']}
)
best_obj.save('meteo/best_model.pkl')
print(f" karachi_aqi_om_production v{version} ({best_name})")

print(f"\nTraining complete!")
print(f"   Best model: {best_name}")
print(f"   MAE: {best_mae:.4f}")
print(f"   Improvement: {((persist_mae-best_mae)/persist_mae*100):.1f}%")