import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/soccer_spain_la_liga/odds/"

TEAM_NAME_MAP = {
    "Alaves": "Deportivo Alavés",
    "Deportivo Alaves": "Deportivo Alavés",
    "Getafe": "Getafe CF",
    "Barcelona": "FC Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Real Sociedad": "Real Sociedad",
    "Sevilla": "Sevilla FC",
    "Villarreal": "Villarreal CF",
    "Valencia": "Valencia CF",
    "Osasuna": "CA Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Celta Vigo": "RC Celta de Vigo",
    "Espanyol": "RCD Espanyol",
    "Real Betis": "Real Betis",
    "Santander": "Racing Santander",
    "Deportivo La Coruna": "Deportivo La Coruña",
    "Malaga": "Málaga CF",
    "Elche": "Elche CF",
    "Levante": "Levante UD",
}

def fetch_draftkings_odds_dict():
    """Fetches DraftKings odds and maps them directly to (home_team, away_team) tuples."""
    if not ODDS_API_KEY:
        return {}

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "bookmakers": "draftkings",
        "oddsFormat": "decimal",
    }

    try:
        res = requests.get(BASE_URL, params=params, timeout=8)
        res.raise_for_status()
        data = res.json()

        odds_map = {}
        for item in data:
            home_raw = item.get("home_team")
            away_raw = item.get("away_team")

            home_team = TEAM_NAME_MAP.get(home_raw, home_raw)
            away_team = TEAM_NAME_MAP.get(away_raw, away_raw)

            bookmakers = item.get("bookmakers", [])
            dk = next((b for b in bookmakers if b["key"] == "draftkings"), None)
            if not dk and bookmakers:
                dk = bookmakers[0]

            if dk:
                markets = dk.get("markets", [])
                h2h = next((m for m in markets if m["key"] == "h2h"), None)
                if h2h:
                    odds_h, odds_d, odds_a = None, None, None
                    for outcome in h2h.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        if name == home_raw:
                            odds_h = price
                        elif name == away_raw:
                            odds_a = price
                        elif name == "Draw":
                            odds_d = price

                    odds_map[(home_team, away_team)] = {
                        "odds_h": odds_h,
                        "odds_d": odds_d,
                        "odds_a": odds_a,
                    }

        return odds_map
    except Exception as e:
        print(f"Error fetching DraftKings odds: {e}")
        return {}