import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os, json, joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
fs = project.get_feature_store()
fg = fs.get_feature_group("weather_pollution_features", version=1)
df = fg.read()

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
df['hour_of_day'] = df['event_timestamp'].dt.hour
df['day_of_week'] = df['event_timestamp'].dt.dayofweek
df['month'] = df['event_timestamp'].dt.month
df['year'] = df['event_timestamp'].dt.year
df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)
df['is_dust_event'] = ((df['pm10'] > 100) & (df['pm2_5'] < 50)).astype(int)
df['is_clean_month'] = (df['month'] == 9).astype(int)

feature_cols = ['pm2_5', 'pm10', 'so2', 'co', 'no2',
                'hour_of_day', 'day_of_week', 'month', 'year',
                'is_winter', 'is_dust_event', 'is_clean_month']

train = df[df['year'] <= 2024]
test = df[df['year'] >= 2025]
X_train = train[feature_cols].fillna(0)
y_train = train['aqi_is_poor']
X_test = test[feature_cols].fillna(0)
y_test = test['aqi_is_poor']

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=50, max_depth=4, random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, num_leaves=15, random_state=42, verbose=-1),
    'MLP_NeuralNet': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

results = {}
best_model = None
best_f1 = 0
best_name = ""
best_good_recall = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    good_rec = ((y_pred == 0) & (y_test == 0)).sum() / (y_test == 0).sum()
    poor_rec = ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    
    results[name] = {
        'f1_score': round(f1, 4),
        'good_recall': round(good_rec, 4),
        'poor_recall': round(poor_rec, 4)
    }

    
    print(f"  {name:20s} | F1: {f1:.4f}")
best_model = models['XGBoost']
best_name = 'XGBoost'
best_f1 = results['XGBoost']['f1_score']
best_good_recall = results['XGBoost']['good_recall']
print(f"\nBest: {best_name} (F1: {best_f1:.4f})")

print("registering models in Hopsworks model registry")
mr = project.get_model_registry()
version = int(datetime.now().strftime('%Y%m%d'))

for name, model_obj in models.items():
    joblib.dump(model_obj, f'{name}.pkl')
    try:
        old = mr.get_model(f"karachi_aqi_{name}", version=version)
        old.delete()
    except:
        pass
    reg_obj = mr.sklearn.create_model(
        name=f"karachi_aqi_{name}",
        version=version,
        metrics=results[name]
    )
    reg_obj.save(f'{name}.pkl')
    print(f"karachi_aqi_{name} v{version}")

try:
    old = mr.get_model("karachi_aqi_production", version=version)
    old.delete()
    print("  🗑️ Deleted old production model")
except:
    pass


joblib.dump(best_model, 'best_model.pkl')
best_obj = mr.sklearn.create_model(
    name="karachi_aqi_production",
    version=version,
    description=f"Production model: {best_name}, F1: {best_f1:.4f}",
    metrics={'f1_score': best_f1}
)
best_obj.save('best_model.pkl')
print(f"karachi_aqi_production v{version} ({best_name})")