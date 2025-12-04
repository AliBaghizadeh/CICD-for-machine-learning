"""
Weather API module for fetching current temperature data.
"""
import os
import requests


def fetch_current_temperature(country_id: str):
    """
    Fetch current temperature for a given country.
    
    Args:
        country_id: Country code (e.g., 'DE', 'FR', 'AT')
    
    Returns:
        dict with keys: temperature, city, description
        or None if fetch fails
    """
    # City mapping
    city_map = {
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
    
    city = city_map.get(country_id, "Berlin")
    
    # Try to get API key from environment
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        # Return default values if no API key
        return {
            "temperature": 15.0,
            "city": city,
            "description": "Clear sky (default)"
        }
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "city": city,
            "description": data["weather"][0]["description"]
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        # Return default on error
        return {
            "temperature": 15.0,
            "city": city,
            "description": "Clear sky (default)"
        }
