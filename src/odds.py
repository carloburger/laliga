# src/odds_service.py
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/soccer_spain_la_liga/odds/"

# Comprehensive team mapping between The Odds API and dataset names
TEAM_NAME_MAP = {
    "Barcelona": "FC Barcelona",
    "FC Barcelona": "FC Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Atlético Madrid": "Atletico Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Athletic Club": "Athletic Club",
    "Real Sociedad": "Real Sociedad",
    "Sevilla": "Sevilla FC",
    "Sevilla FC": "Sevilla FC",
    "Villarreal": "Villarreal CF",
    "Villarreal CF": "Villarreal CF",
    "Valencia": "Valencia CF",
    "Valencia CF": "Valencia CF",
    "Getafe": "Getafe CF",
    "Getafe CF": "Getafe CF",
    "Osasuna": "CA Osasuna",
    "CA Osasuna": "CA Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Celta Vigo": "RC Celta de Vigo",
    "RC Celta de Vigo": "RC Celta de Vigo",
    "Alaves": "Deportivo Alavés",
    "Deportivo Alaves": "Deportivo Alavés",
    "Deportivo Alavés": "Deportivo Alavés",
    "Mallorca": "RCD Mallorca",
    "RCD Mallorca": "RCD Mallorca",
    "Girona": "Girona FC",
    "Girona FC": "Girona FC",
    "Las Palmas": "UD Las Palmas",
    "UD Las Palmas": "UD Las Palmas",
    "Espanyol": "RCD Espanyol",
    "RCD Espanyol": "RCD Espanyol",
    "Leganes": "CD Leganés",
    "CD Leganés": "CD Leganés",
    "Valladolid": "Real Valladolid",
    "Real Valladolid": "Real Valladolid",
    "Real Betis": "Real Betis",
}

def fetch_draftkings_la_liga_odds():
    if not ODDS_API_KEY:
        return []

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

        fixtures = []
        for idx, item in enumerate(data):
            home_raw = item.get("home_team")
            away_raw = item.get("away_team")

            home_team = TEAM_NAME_MAP.get(home_raw, home_raw)
            away_team = TEAM_NAME_MAP.get(away_raw, away_raw)

            odds_h, odds_d, odds_a = None, None, None

            bookmakers = item.get("bookmakers", [])
            dk = next((b for b in bookmakers if b["key"] == "draftkings"), None)
            if not dk and bookmakers:
                dk = bookmakers[0]

            if dk:
                markets = dk.get("markets", [])
                h2h = next((m for m in markets if m["key"] == "h2h"), None)
                if h2h:
                    for outcome in h2h.get("outcomes", []):
                        name = outcome.get("name")
                        price = outcome.get("price")
                        if name == home_raw:
                            odds_h = price
                        elif name == away_raw:
                            odds_a = price
                        elif name == "Draw":
                            odds_d = price

            fixtures.append({
                "id": idx + 1,
                "date": item.get("commence_time", "")[:10],
                "home": home_team,
                "away": away_team,
                "odds_h": odds_h,
                "odds_d": odds_d,
                "odds_a": odds_a,
            })

        return fixtures
    except Exception as e:
        print(f"Error fetching live odds: {e}")
        return []