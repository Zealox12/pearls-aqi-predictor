# test_api_depth.py
import requests, os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv("API_KEY")
lat, lon = 24.8607, 67.0011
end_date = datetime.now()

print("Testing how far back OpenWeather free API goes...\n")

for months_back in range(1,100):
    start_date = end_date - timedelta(days=30 * months_back)
    end_chunk = end_date - timedelta(days=30 * (months_back - 1))
    
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/air_pollution/history",
        params={
            'lat': lat, 'lon': lon,
            'start': int(start_date.timestamp()),
            'end': int(end_chunk.timestamp()),
            'appid': API_KEY
        }
    )
    
    if r.status_code == 200:
        records = len(r.json().get('list', []))
        print(f"✅ {months_back} months back ({start_date.strftime('%b %Y')}): {records} records")
    else:
        print(f"❌ {months_back} months back ({start_date.strftime('%b %Y')}): Error {r.status_code}")