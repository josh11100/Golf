"""
Weather enrichment.

Two sources, used for what each is good at:
- Open-Meteo archive API (free, no key): HISTORICAL daily weather for past
  tournament rounds — this is what the ML features use.
- OpenWeather (your API key in .env): CURRENT conditions + 5-day forecast —
  used by the dashboard for upcoming-event context.

Usage:
    python features/weather.py backfill   # fill temp/wind/precip on stg_rounds
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, OPENWEATHER_API_KEY

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENWEATHER_CURRENT = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"


def fetch_historical_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Daily historical weather from Open-Meteo (no key needed)."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,wind_speed_10m_max,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    r = httpx.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({
        "round_date": d["time"],
        "temp_f": d["temperature_2m_max"],
        "wind_mph": d["wind_speed_10m_max"],
        "precip_in": d["precipitation_sum"],
    })


def fetch_current(lat: float, lon: float) -> dict:
    """Current conditions from OpenWeather (uses your key)."""
    r = httpx.get(OPENWEATHER_CURRENT, params={
        "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "imperial",
    }, timeout=20)
    r.raise_for_status()
    j = r.json()
    return {
        "temp_f": j["main"]["temp"],
        "wind_mph": j["wind"]["speed"],
        "conditions": j["weather"][0]["description"],
    }


def backfill_rounds():
    """Fill temp_f / wind_mph / precip_in on stg_rounds from Open-Meteo,
    one API call per tournament (covers all its round dates)."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        tourneys = pd.read_sql(text("""
            SELECT t.tournament_id, t.latitude, t.longitude,
                   MIN(r.round_date) AS start_d, MAX(r.round_date) AS end_d
            FROM stg_tournaments t
            JOIN stg_rounds r ON r.tournament_id = t.tournament_id
            WHERE r.temp_f IS NULL
            GROUP BY t.tournament_id, t.latitude, t.longitude
        """), conn)

    if tourneys.empty:
        print("Nothing to backfill — all rounds already have weather.")
        return

    engine2 = create_engine(DATABASE_URL)
    for _, t in tourneys.iterrows():
        try:
            wx = fetch_historical_daily(t.latitude, t.longitude, t.start_d, t.end_d)
        except Exception as e:
            print(f"  tournament {t.tournament_id}: weather fetch failed ({e})")
            continue
        with engine2.begin() as conn:
            for _, w in wx.iterrows():
                conn.execute(text("""
                    UPDATE stg_rounds
                    SET temp_f = :temp, wind_mph = :wind, precip_in = :precip
                    WHERE tournament_id = :tid AND round_date = :d
                """), dict(temp=w.temp_f, wind=w.wind_mph,
                           precip=w.precip_in, tid=int(t.tournament_id),
                           d=w.round_date))
        print(f"  tournament {t.tournament_id}: {len(wx)} days backfilled")
        time.sleep(0.5)  # be polite

    print("Backfill complete. Re-run etl/load.py marts or features next.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill_rounds()
    else:
        print(__doc__)
