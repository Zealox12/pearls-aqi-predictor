"""
Karachi AQI Forecast Dashboard
Run with: streamlit run dashboard.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import requests
from collections import defaultdict

import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inline CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* Header */
    .aqi-header {
        background: linear-gradient(135deg, #0f1923 0%, #162032 60%, #1a2a40 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.07);
        position: relative;
        overflow: hidden;
    }
    .aqi-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(255,140,0,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .aqi-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        margin: 0;
    }
    .aqi-subtitle {
        color: rgba(255,255,255,0.45);
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.35rem;
        letter-spacing: 0.05em;
    }

    /* Day cards */
    .day-card {
        background: #111b27;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        height: 100%;
        transition: border-color 0.2s;
    }
    .day-card:hover { border-color: rgba(255,255,255,0.18); }
    .day-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e8eaf0;
    }
    .day-date {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        color: rgba(255,255,255,0.35);
        margin-top: 2px;
    }
    .severity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0.6rem 0;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
    }
    .stat-value {
        color: #e8eaf0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.88rem;
    }

    /* Hourly timeline */
    .hour-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.8rem;
    }
    .hour-label {
        width: 52px;
        font-family: 'JetBrains Mono', monospace;
        color: rgba(255,255,255,0.35);
        font-size: 0.75rem;
        flex-shrink: 0;
    }
    .hour-bar-wrap { flex: 1; }
    .aqi-val {
        width: 28px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        flex-shrink: 0;
    }
    .ow-val {
        width: 60px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.3);
        flex-shrink: 0;
    }

    /* Current conditions */
    .cond-card {
        background: linear-gradient(135deg, #0e2a1a 0%, #0f1f2e 100%);
        border: 1px solid rgba(72,230,120,0.2);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .cond-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(72,230,120,0.7);
        margin-bottom: 0.8rem;
    }
    .cond-grid {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .cond-item { text-align: center; }
    .cond-num {
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #e8eaf0;
        line-height: 1;
    }
    .cond-label {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.35);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Streamlit overrides */
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── AQI helpers ──────────────────────────────────────────────────────────────
AQI_LEVELS = {
    1: {"label": "Good",      "emoji": "🟢", "color": "#22c55e", "bg": "rgba(34,197,94,0.15)",  "text": "#22c55e"},
    2: {"label": "Fair",      "emoji": "🟡", "color": "#84cc16", "bg": "rgba(132,204,22,0.15)", "text": "#84cc16"},
    3: {"label": "Poor",      "emoji": "🟠", "color": "#f97316", "bg": "rgba(249,115,22,0.15)", "text": "#f97316"},
    4: {"label": "Very Poor", "emoji": "🔴", "color": "#ef4444", "bg": "rgba(239,68,68,0.15)",  "text": "#ef4444"},
    5: {"label": "Hazardous", "emoji": "🟣", "color": "#7c3aed", "bg": "rgba(124,58,237,0.15)", "text": "#a78bfa"},
}



def aqi_class_to_style(aqi_class) -> dict:
    try:
        cls = int(round(float(aqi_class)))
        cls = max(1, min(5, cls))
    except (TypeError, ValueError):
        cls = 3
    return AQI_LEVELS[cls]


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_forecast():
    """Load forecast from GitHub raw URL (works on Streamlit Cloud)"""
    url = "https://raw.githubusercontent.com/Zealox12/pearls-aqi-predictor/main/meteo/aqi_forecast.json"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Could not fetch forecast (HTTP {response.status_code})")
            return {}
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not connect to GitHub: {e}")
        return {}
    except json.JSONDecodeError as e:
        st.warning(f"Invalid JSON in forecast file: {e}")
        return {}


# Sample current conditions (replace with real sensor feed if available)
CURRENT_CONDITIONS = {
    "temperature": 34.2,
    "humidity": 71,
    "pm25": 87.4,
    "pm10": 142.1,
    "aqi_class": 4,
}

# Replace the hardcoded CURRENT_CONDITIONS with:
def get_current_conditions(data):
    hourly = data.get("hourly", [])
    if not hourly:
        return {"pm25": "N/A", "pm10": "N/A", "aqi_class": 3}
    latest = hourly[0]
    aqi_val = latest.get("aqi", 60)
    aqi_class = (1 if aqi_val <= 20 else 2 if aqi_val <= 40 else
                 3 if aqi_val <= 60 else 4 if aqi_val <= 80 else 5)
    return {
        "pm25": latest.get("pm25", "N/A"),
        "pm10": latest.get("pm10", "N/A"),
        "aqi_class": aqi_class
    }

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    auto_refresh = st.toggle("Auto-refresh (hourly)", value=False)
    refresh_interval = st.selectbox(
        "Refresh interval",
        options=[15, 30, 60],
        index=2,
        format_func=lambda x: f"Every {x} min",
        disabled=not auto_refresh,
    )
    st.divider()
    st.markdown(
        "<small style='color:rgba(255,255,255,0.3)'>Data source: aqi_forecast.json</small>",
        unsafe_allow_html=True,
    )

# ── Load data ────────────────────────────────────────────────────────────────
data = load_forecast()

# ── Header ───────────────────────────────────────────────────────────────────
generated_at = data.get("generated_at", "—")
try:
    gen_dt = datetime.fromisoformat(generated_at)
    gen_str = gen_dt.strftime("%A, %d %B %Y · %H:%M")
except (ValueError, TypeError):
    gen_str = generated_at

col_hdr, col_btn = st.columns([5, 1])
with col_hdr:
    st.markdown(
        f"""
        <div class="aqi-header">
            <div class="aqi-title">🌫️ Karachi AQI Forecast</div>
            <div class="aqi-subtitle">Generated: {gen_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── Current conditions ───────────────────────────────────────────────────────
cc = get_current_conditions(data)
aqi_style = aqi_class_to_style(cc["aqi_class"])

st.markdown(
    f"""
    <div class="cond-card">
        <div class="cond-title">📍 Current Conditions — Karachi</div>
        <div class="cond-grid">
            <div class="cond-item">
                <div class="cond-num">{cc['pm25']}</div>
                <div class="cond-label">PM2.5 µg/m³</div>
            </div>
            <div class="cond-item">
                <div class="cond-num">{cc['pm10']}</div>
                <div class="cond-label">PM10 µg/m³</div>
            </div>
            <div class="cond-item">
                <div class="cond-num" style="color:{aqi_style['color']}">{aqi_style['emoji']} {aqi_style['label']}</div>
                <div class="cond-label">AQI Level</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 3-Day summary cards ───────────────────────────────────────────────────────
# Replace the daily cards section with:
daily = data.get("daily", {})
hourly = data.get("hourly", [])

if hourly:
    # Group hourly into days
    days = defaultdict(list)
    for h in hourly:
        day_key = h['time'][:10]  # Extract date from ISO timestamp
        days[day_key].append(h)
    
    st.markdown("#### 📅 3-Day Forecast")
    cols = st.columns(min(len(days), 3))
    
    for idx, (date_key, hours) in enumerate(list(days.items())[:3]):
        aqi_vals = [h['aqi'] for h in hours]
        avg_aqi = sum(aqi_vals) / len(aqi_vals)
        max_aqi = max(aqi_vals)
        aqi_class = 1 if max_aqi <= 20 else 2 if max_aqi <= 40 else 3 if max_aqi <= 60 else 4 if max_aqi <= 80 else 5
        sev_style = AQI_LEVELS[aqi_class]
        severity = sev_style['label']
        poor_count = sum(1 for a in aqi_vals if a > 60)
        
        with cols[idx]:
            st.markdown(
                f"""
                <div class="day-card">
                    <div class="day-name">{date_key}</div>
                    <div>
                        <span class="severity-badge"
                            style="background:{sev_style['bg']};color:{sev_style['text']};">
                            {sev_style['emoji']} {severity}
                        </span>
                    </div>
                    <div class="stat-row">
                        <span>Avg AQI</span>
                        <span class="stat-value">{avg_aqi:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span>Max AQI</span>
                        <span class="stat-value">{max_aqi:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span>Poor Hours</span>
                        <span class="stat-value">{poor_count}/{len(hours)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Hourly breakdown ──────────────────────────────────────────────────────────

def check_alerts(hourly_data):
    alerts = []
    for h in hourly_data:
        if h['aqi'] > 80:
            alerts.append({'time': h['time'], 'aqi': h['aqi'], 'level': '🔴 Hazardous'})
        elif h['aqi'] > 60:
            alerts.append({'time': h['time'], 'aqi': h['aqi'], 'level': '🟠 Poor'})
    return alerts

alerts = check_alerts(hourly)
if alerts:
    st.error(f"⚠️ {len(alerts)} critical AQI hours detected in next 72 hours!")
    for alert in alerts[:5]:
        st.warning(f"  {alert['time'][:16]} - AQI: {alert['aqi']:.1f} ({alert['level']})")
    
if hourly:
    st.markdown("#### 🕐 Hourly Breakdown")
    from collections import defaultdict
    hourly_by_date = defaultdict(list)
    for h in hourly:
        day_key = h['time'][:10]
        hourly_by_date[day_key].append(h)
    
    for date_key, hours in list(hourly_by_date.items())[:3]:
        poor_count = sum(1 for h in hours if h.get('aqi', 60) > 60)
        label = f"{date_key} · {poor_count}/{len(hours)} poor hours"
        
        with st.expander(label, expanded=(date_key == list(hourly_by_date.keys())[0])):
            for h in sorted(hours, key=lambda x: x.get('time', '')):  # Every 3 hours
                hr = h['time'][11:16]  # Extract HH:MM from ISO
                aqi_val = h.get('aqi', 60)
                aqi_class = 1 if aqi_val <= 20 else 2 if aqi_val <= 40 else 3 if aqi_val <= 60 else 4 if aqi_val <= 80 else 5
                style = AQI_LEVELS[aqi_class]
                st.markdown(f"{style['emoji']} **{hr}** — AQI: {aqi_val:.1f} ({style['label']})")
elif not data:
    st.info("ℹ️ Place your `aqi_forecast.json` file in the same directory as `dashboard.py`, then refresh.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small style='color:rgba(255,255,255,0.2);font-family:JetBrains Mono,monospace;'>"
    "Karachi AQI Forecast · Data refreshes from aqi_forecast.json · "
    f"Dashboard loaded at {datetime.now().strftime('%H:%M:%S')}"
    "</small>",
    unsafe_allow_html=True,
)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_interval * 60}">',
        unsafe_allow_html=True
    )