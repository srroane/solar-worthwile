from __future__ import annotations

import pandas as pd
import requests
import streamlit as st


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOLAR_URL = "https://api.forecast.solar/estimate"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_city(city_name: str) -> dict:
    params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
    response = requests.get(GEOCODING_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    results = payload.get("results")
    if not results:
        raise ValueError(f"No location found for '{city_name}'.")

    top = results[0]
    return {
        "name": top.get("name", city_name),
        "country": top.get("country", ""),
        "latitude": float(top["latitude"]),
        "longitude": float(top["longitude"]),
    }


@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_weather_forecast(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,cloud_cover,shortwave_radiation",
        "forecast_days": days,
        "timezone": "auto",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=15)
    response.raise_for_status()
    hourly = response.json()["hourly"]

    weather = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "ghi": hourly["shortwave_radiation"],
            "temperature_c": hourly["temperature_2m"],
            "cloud_cover_pct": hourly["cloud_cover"],
        }
    ).set_index("timestamp")

    return weather.fillna(0.0).astype(float)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_solar_production(
    latitude: float,
    longitude: float,
    declination: float,
    azimuth: float,
    kwp: float,
) -> pd.DataFrame:
    url = f"{SOLAR_URL}/{latitude}/{longitude}/{declination}/{azimuth}/{kwp}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    daily_wh = response.json()["result"]["watt_hours_day"]

    return pd.DataFrame(
        [
            {"date": pd.to_datetime(day), "production_kwh": wh / 1000.0}
            for day, wh in daily_wh.items()
        ]
    ).set_index("date")
