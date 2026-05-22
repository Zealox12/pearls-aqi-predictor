import requests

lat, lon = 24.8607, 67.0011

# Test historical air quality availability
for months_back in [1, 6, 12, 24, 36, 48, 60,100, 150, 160]:
    end_date = "2026-05-22"
    start_date = f"2026-{5-months_back:02d}-22" if months_back < 5 else f"{2026-months_back//12}-{(12-months_back%12):02d}-22"
    
    # Simpler: test a specific date
    from datetime import datetime, timedelta
    test_date = datetime.now() - timedelta(days=30 * months_back)
    
    r = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            'latitude': lat, 'longitude': lon,
            'start_date': test_date.strftime('%Y-%m-%d'),
            'end_date': test_date.strftime('%Y-%m-%d'),
            'hourly': 'pm2_5,pm10'
        }
    )
    
    if r.status_code == 200:
        data = r.json()
        records = len(data.get('hourly', {}).get('time', []))
        print(f"✅ {months_back} months back ({test_date.strftime('%b %Y')}): {records} records")
    else:
        print(f"❌ {months_back} months back ({test_date.strftime('%b %Y')}): Error {r.status_code}")