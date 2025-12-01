"""
OpenWeatherMap API integration for fetching real-time temperature data.
"""

import os
import requests
from typing import Optional, Dict

# Country code to capital city mapping
COUNTRY_CITIES = {
    "AT": "Vienna",
    "DE": "Berlin",
    "FR": "Paris",
    "IT": "Rome",
    "BE": "Brussels",
    "CH": "Zurich",
    "NL": "Amsterdam",
    "PL": "Warsaw",
    "CZ": "Prague",
    "ES": "Madrid",
}


def fetch_current_temperature(country_code: str) -> Optional[Dict[str, any]]:
    """
    Fetch current temperature for a given country using OpenWeatherMap API.

    Args:
        country_code: Two-letter country code (e.g., 'DE', 'FR')

    Returns:
        Dictionary with 'temperature' (Celsius), 'city', and 'description',
        or None if fetch fails
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        print("⚠️ OPENWEATHER_API_KEY not found in environment variables")
        return None

    city = COUNTRY_CITIES.get(country_code)
    if not city:
        print(f"⚠️ Unknown country code: {country_code}")
        return None

    try:
        # OpenWeatherMap API endpoint
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()

        return {
            "temperature": round(data["main"]["temp"], 1),
            "city": city,
            "description": data["weather"][0]["description"],
            "country": country_code,
        }

    except requests.exceptions.Timeout:
        print(f"⚠️ Timeout fetching weather for {city}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching weather for {city}: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"⚠️ Error parsing weather data for {city}: {e}")
        return None
