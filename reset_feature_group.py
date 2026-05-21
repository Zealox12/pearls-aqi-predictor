import sys, types
m = types.ModuleType('pyjks')
sys.modules['pyjks'] = m

import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv('API_KEY_HS'),
    project="Pearls_AQI_Predictor12",
    host="eu-west.cloud.hopsworks.ai"
)
fs = project.get_feature_store()

try:
    fg = fs.get_feature_group("weather_pollution_features", version=1)
    fg.delete()
    print("Old feature group deleted")
except:
    print("No feature group to delete")

print("Ready for fresh data")