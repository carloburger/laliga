import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure src directory is in Python path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Internal Module Imports
from features import prepare_dataset
from train import fine_tune_and_train
from odds import fetch_draftkings_odds_dict

# Load environment configuration
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
EST_TZ = ZoneInfo("America/New_York")

# ==========================================
# 1. AMERICAN ODDS CONVERSION UTILITIES
# ==========================================

def decimal_to_american(decimal_odds: float) -> str:
    """Converts decimal odds (e.g. 2.10, 1.80) to American odds format (+110, -125)."""
    if not decimal_odds or decimal_odds <= 1.0:
        return "+100"
    if decimal_odds >= 2.0:
        val = int(round((decimal_odds - 1.0) * 100))
        return f"+{val}"
    else:
        val = int(round(-100.0 / (decimal_odds - 1.0)))
        return f"{val}"


def american_to_decimal(american_input: Union[str, float, int]) -> float:
    """Converts American odds (+150, -110) or raw decimal numeric inputs to decimal odds."""
    if isinstance(american_input, (float, int)) and 1.0 < american_input < 50.0:
        return float(american_input)

    clean_str = str(american_input).replace("+", "").strip()
    try:
        val = float(clean_str)
        if val > 0:
            return (val / 100.0) + 1.0
        elif val < 0:
            return (100.0 / abs(val)) + 1.0
        return 2.0
    except ValueError:
        return 2.0

# ==========================================
# 2. TEAM MAPS & NORMALIZATION LOGIC
# ==========================================

LA_LIGA_TEAMS = [
    "Athletic Club",
    "Atletico Madrid",
    "CA Osasuna",
    "Deportivo Alavés",
    "Deportivo La Coruña",
    "Elche CF",
    "FC Barcelona",
    "Getafe CF",
    "Levante UD",
    "Málaga CF",
    "Racing Santander",
    "Rayo Vallecano",
    "RC Celta de Vigo",
    "RCD Espanyol",
    "Real Betis",
    "Real Madrid",
    "Real Sociedad",
    "Sevilla FC",
    "Valencia CF",
    "Villarreal CF",
]

DATASET_TEAM_MAP = {
    "RCD Espanyol": ["RCD Espanyol", "Espanyol", "RCD Espanyol de Barcelona"],
    "FC Barcelona": ["FC Barcelona", "Barcelona", "Barca"],
    "Real Madrid": ["Real Madrid", "R Madrid"],
    "Atletico Madrid": ["Atletico Madrid", "Atl Madrid", "Atlético Madrid", "Ath Madrid"],
    "Athletic Club": ["Athletic Club", "Ath Bilbao", "Athletic Bilbao"],
    "Real Sociedad": ["Real Sociedad", "R Sociedad"],
    "Sevilla FC": ["Sevilla FC", "Sevilla"],
    "Villarreal CF": ["Villarreal CF", "Villarreal"],
    "Valencia CF": ["Valencia CF", "Valencia"],
    "Getafe CF": ["Getafe CF", "Getafe"],
    "CA Osasuna": ["CA Osasuna", "Osasuna"],
    "Rayo Vallecano": ["Rayo Vallecano", "Rayo"],
    "RC Celta de Vigo": ["RC Celta de Vigo", "Celta Vigo", "Celta"],
    "Deportivo Alavés": ["Deportivo Alavés", "Deportivo Alaves", "Alaves"],
    "Real Betis": ["Real Betis", "Betis", "Real Betis Balompié"],
    "Racing Santander": ["Racing Santander", "Racing", "Santander", "Real Racing Club"],
    "Deportivo La Coruña": ["Deportivo La Coruña", "Deportivo La Coruna", "Deportivo", "Real Club Deportivo de A Coruña"],
    "Málaga CF": ["Málaga CF", "Malaga CF", "Malaga"],
    "Elche CF": ["Elche CF", "Elche"],
    "Levante UD": ["Levante UD", "Levante"],
}

PROMOTED_TEAMS = ["Racing Santander", "Deportivo La Coruña", "Málaga CF", "Elche CF", "Levante UD"]


def normalize_team_name(raw_name: str) -> str:
    """Standardizes raw team strings with explicit priority check for Espanyol."""
    if not raw_name:
        return "Unknown Team"
    clean_raw = raw_name.strip()
    clean_lower = clean_raw.lower()

    # Priority check for Espanyol to prevent 'Barcelona' substring collision
    if "espanyol" in clean_lower:
        return "RCD Espanyol"

    for canonical_name, variants in DATASET_TEAM_MAP.items():
        if clean_raw in variants:
            return canonical_name
        if any(v.lower() in clean_lower for v in variants):
            return canonical_name

    return clean_raw

# ==========================================
# 3. DYNAMIC FIXTURE PIPELINE
# ==========================================

def fetch_fixtures_football_data_api(matchday: int) -> List[Dict[str, Any]]:
    """Tier 1: Fetches official matchday fixtures from Football-Data.org API."""
    if not FOOTBALL_DATA_API_KEY:
        return []

    url = "https://api.football-data.org/v4/competitions/PD/matches"
    headers = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
    params = {"matchday": matchday}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=8)
        if res.status_code != 200:
            return []

        data = res.json()
        matches = data.get("matches", [])
        results = []

        for idx, m in enumerate(matches, 1):
            utc_str = m.get("utcDate")
            if utc_str:
                dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00")).astimezone(EST_TZ)
                date_str = dt.strftime("%b %d, %I:%M %p EST")
            else:
                date_str = "TBD"

            results.append({
                "id": m.get("id", idx),
                "matchday": matchday,
                "date": date_str,
                "home": normalize_team_name(m.get("homeTeam", {}).get("name", "")),
                "away": normalize_team_name(m.get("awayTeam", {}).get("name", "")),
                "odds_h": None,
                "odds_d": None,
                "odds_a": None,
            })
        return results
    except Exception as e:
        print(f"[API ERROR] Football-Data API failed: {e}")
        return []


def scrape_laliga_matchday_web(matchday: int) -> List[Dict[str, Any]]:
    """Tier 2: Live web scraper that extracts matchday schedule from WorldFootball.net."""
    url = f"https://www.worldfootball.net/schedule/esp-primera-division-2026-2027-spieltag/{matchday}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code != 200:
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", {"class": "standard_tabelle"})
        if not table:
            return []

        fixtures = []
        match_id = 1
        current_date_str = "Aug 15"

        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            date_col = cols[0].get_text(strip=True)
            if date_col and len(date_col) > 3:
                current_date_str = date_col

            time_col = cols[1].get_text(strip=True) or "3:00 PM EST"
            home_raw = cols[2].get_text(strip=True)
            away_raw = cols[4].get_text(strip=True)

            if home_raw and away_raw and home_raw != away_raw:
                fixtures.append({
                    "id": (matchday * 100) + match_id,
                    "matchday": matchday,
                    "date": f"{current_date_str}, {time_col}",
                    "home": normalize_team_name(home_raw),
                    "away": normalize_team_name(away_raw),
                    "odds_h": None,
                    "odds_d": None,
                    "odds_a": None,
                })
                match_id += 1

        return fixtures
    except Exception as e:
        print(f"[SCRAPER ERROR] Web scraping failed for matchday {matchday}: {e}")
        return []


def get_dynamic_matchday_fixtures(matchday: int) -> List[Dict[str, Any]]:
    """Executes dynamic fixture pipeline: API -> Scraper."""
    fixtures = fetch_fixtures_football_data_api(matchday)
    if not fixtures:
        fixtures = scrape_laliga_matchday_web(matchday)
    return fixtures

# ==========================================
# 4. PYDANTIC SCHEMAS
# ==========================================

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    odds_h: Union[str, float, int] = Field(..., description="American odds string or numeric decimal odds")
    odds_d: Union[str, float, int] = Field(..., description="American odds string or numeric decimal odds")
    odds_a: Union[str, float, int] = Field(..., description="American odds string or numeric decimal odds")


class FairOdds(BaseModel):
    odds_h_american: str
    odds_d_american: str
    odds_a_american: str
    odds_h: float
    odds_d: float
    odds_a: float


class SimulationResponse(BaseModel):
    lambda_home: float
    lambda_away: float
    sims: Dict[str, Any]
    fair_odds: FairOdds
    suggestions: List[str]

# ==========================================
# 5. PREDICTIVE EXPECTED GOALS (xG) ENGINE
# ==========================================

def get_team_matches(df: pd.DataFrame, team_ui_name: str, is_home: bool) -> pd.DataFrame:
    """Queries team matches using dataset variants."""
    variants = DATASET_TEAM_MAP.get(team_ui_name, [team_ui_name])
    col = "HomeTeam" if is_home else "AwayTeam"
    return df[df[col].isin(variants)]


def predict_lambdas(
    home_team: str, 
    away_team: str, 
    df: pd.DataFrame, 
    home_model=None, 
    away_model=None
) -> Tuple[float, float]:
    """Calculates expected goals (lambdas) using historical decay, promoted team baselines,

    head-to-head records, and LightGBM model blending.
    """
    default_home_xg, default_away_xg = 1.35, 1.15

    if df is None or df.empty:
        return default_home_xg, default_away_xg

    try:
        df_calc = df.copy()

        # Exponential Time Decay (~500 day half-life)
        if "Date" in df_calc.columns:
            df_calc["Date"] = pd.to_datetime(df_calc["Date"], errors="coerce")
            max_date = df_calc["Date"].max()
            df_calc["days_old"] = (max_date - df_calc["Date"]).dt.days.fillna(1000)
            df_calc["weight"] = np.exp(-df_calc["days_old"] / 500)
        else:
            df_calc["weight"] = 1.0

        league_home_avg = np.average(df_calc["FTHG"], weights=df_calc["weight"])
        league_away_avg = np.average(df_calc["FTAG"], weights=df_calc["weight"])

        h_df = get_team_matches(df_calc, home_team, is_home=True)
        a_df = get_team_matches(df_calc, away_team, is_home=False)

        # Home Team Ratings
        if not h_df.empty and h_df["weight"].sum() > 0.5:
            h_att = np.average(h_df["FTHG"], weights=h_df["weight"]) / league_home_avg
            h_def = np.average(h_df["FTAG"], weights=h_df["weight"]) / league_away_avg
        else:
            h_att = 0.88 if home_team in PROMOTED_TEAMS else 1.0
            h_def = 1.15 if home_team in PROMOTED_TEAMS else 1.0

        # Away Team Ratings
        if not a_df.empty and a_df["weight"].sum() > 0.5:
            a_att = np.average(a_df["FTAG"], weights=a_df["weight"]) / league_away_avg
            a_def = np.average(a_df["FTHG"], weights=a_df["weight"]) / league_home_avg
        else:
            a_att = 0.82 if away_team in PROMOTED_TEAMS else 1.0
            a_def = 1.20 if away_team in PROMOTED_TEAMS else 1.0

        lambda_h = h_att * a_def * league_home_avg
        lambda_a = a_att * h_def * league_away_avg

        # Head-to-Head Adjustments
        home_variants = DATASET_TEAM_MAP.get(home_team, [home_team])
        away_variants = DATASET_TEAM_MAP.get(away_team, [away_team])

        h2h = df_calc[
            ((df_calc["HomeTeam"].isin(home_variants)) & (df_calc["AwayTeam"].isin(away_variants))) |
            ((df_calc["HomeTeam"].isin(away_variants)) & (df_calc["AwayTeam"].isin(home_variants)))
        ].sort_values(by="Date", ascending=False).head(8)

        total_w = 0.0
        h2h_home_share, h2h_away_share = 0.5, 0.5

        if not h2h.empty:
            home_h2h_goals, away_h2h_goals = 0.0, 0.0
            for _, row in h2h.iterrows():
                w = row["weight"]
                if row["HomeTeam"] in home_variants:
                    home_h2h_goals += row["FTHG"] * w
                    away_h2h_goals += row["FTAG"] * w
                else:
                    home_h2h_goals += row["FTAG"] * w
                    away_h2h_goals += row["FTHG"] * w
                total_w += w

            if total_w > 0 and (home_h2h_goals + away_h2h_goals) > 0:
                tot = home_h2h_goals + away_h2h_goals
                h2h_home_share = home_h2h_goals / tot
                h2h_away_share = away_h2h_goals / tot
                lambda_h *= (0.80 + 0.40 * h2h_home_share)
                lambda_a *= (0.80 + 0.40 * h2h_away_share)

        # LightGBM Model Integration
        if home_model is not None and away_model is not None:
            try:
                feature_dict = {
                    "H_ATT": h_att, "H_DEF": h_def, "A_ATT": a_att, "A_DEF": a_def,
                    "H2H_WEIGHT": total_w, "league_home_avg": league_home_avg, "league_away_avg": league_away_avg,
                    "h2h_home_share": h2h_home_share, "h2h_away_share": h2h_away_share,
                    "is_home_promoted": 1.0 if home_team in PROMOTED_TEAMS else 0.0,
                    "is_away_promoted": 1.0 if away_team in PROMOTED_TEAMS else 0.0,
                }
                expected_features = getattr(home_model, "feature_name_", [])
                feat_row = pd.DataFrame([{col: feature_dict.get(col, 0.0) for col in expected_features}]) if expected_features else pd.DataFrame([feature_dict])

                ml_lambda_h = float(home_model.predict(feat_row)[0])
                ml_lambda_a = float(away_model.predict(feat_row)[0])
                lambda_h = 0.5 * lambda_h + 0.5 * ml_lambda_h
                lambda_a = 0.5 * lambda_a + 0.5 * ml_lambda_a
            except Exception as e:
                print(f"[WARN] LightGBM inference skipped: {e}")

        lambda_h = float(np.clip(lambda_h, 0.30, 4.20))
        lambda_a = float(np.clip(lambda_a, 0.30, 4.20))
        return round(lambda_h, 2), round(lambda_a, 2)

    except Exception as e:
        print(f"Error computing lambdas for {home_team} vs {away_team}: {e}")
        return default_home_xg, default_away_xg

# ==========================================
# 6. MONTE CARLO POISSON ENGINE
# ==========================================

def simulate_match_poisson(lambda_home: float, lambda_away: float, num_sims: int = 10000, rho: float = -0.13) -> Dict[str, Any]:
    """Runs Monte Carlo simulations using Dixon-Coles adjusted Poisson distribution."""
    home_goals = np.random.poisson(lambda_home, num_sims)
    away_goals = np.random.poisson(lambda_away, num_sims)

    weights = np.ones(num_sims)

    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    weights[mask_00] = max(0.1, 1.0 - (lambda_home * lambda_away * rho))
    weights[mask_10] = max(0.1, 1.0 + (lambda_away * rho))
    weights[mask_01] = max(0.1, 1.0 + (lambda_home * rho))
    weights[mask_11] = max(0.1, 1.0 - rho)
    weights /= np.sum(weights)

    home_win_prob = float(np.sum(weights[home_goals > away_goals]))
    draw_prob = float(np.sum(weights[home_goals == away_goals]))
    away_win_prob = float(np.sum(weights[home_goals < away_goals]))

    return {
        "home_win_prob": round(home_win_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "over_2_5_prob": round(float(np.sum(weights[(home_goals + away_goals) > 2.5])), 4),
        "btts_prob": round(float(np.sum(weights[(home_goals > 0) & (away_goals > 0)])), 4),
    }


def generate_betting_insights(sims: Dict[str, Any], market_h_dec: float, market_d_dec: float, market_a_dec: float) -> List[str]:
    """Detects positive EV betting edges against market prices."""
    insights = []
    implied_h = 1.0 / market_h_dec if market_h_dec > 0 else 0
    implied_d = 1.0 / market_d_dec if market_d_dec > 0 else 0
    implied_a = 1.0 / market_a_dec if market_a_dec > 0 else 0

    edge_h = sims["home_win_prob"] - implied_h
    edge_d = sims["draw_prob"] - implied_d
    edge_a = sims["away_win_prob"] - implied_a

    if edge_h > 0.035:
        insights.append(f"🔥 Positive EV on Home Win: Model gives {(sims['home_win_prob']*100):.1f}% vs Market Implied {(implied_h*100):.1f}% (+{(edge_h*100):.1f}% edge).")
    elif edge_a > 0.035:
        insights.append(f"🔥 Positive EV on Away Win: Model gives {(sims['away_win_prob']*100):.1f}% vs Market Implied {(implied_a*100):.1f}% (+{(edge_a*100):.1f}% edge).")
    elif edge_d > 0.040:
        insights.append(f"⚖️ Value on Draw Market: Model probability {(sims['draw_prob']*100):.1f}% exceeds market pricing.")
    else:
        insights.append("🎯 Market lines are well-calibrated against model predictions.")

    if sims["over_2_5_prob"] > 0.58:
        insights.append(f"⚽ Over 2.5 Goals Lean: High expected goal volume ({(sims['over_2_5_prob']*100):.1f}% probability).")

    return insights

# ==========================================
# 7. FASTAPI APPLICATION & ENDPOINTS
# ==========================================

app = FastAPI(title="La Liga Predictive Betting Engine", version="2.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent.parent / "data" / "laliga_merged_2018_2025.csv"
GLOBAL_DF: pd.DataFrame = None
HOME_MODEL = None
AWAY_MODEL = None


@app.on_event("startup")
def startup_event():
    global GLOBAL_DF, HOME_MODEL, AWAY_MODEL
    print("Loading La Liga historical dataset...")
    if DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
        GLOBAL_DF = prepare_dataset(raw_df)
        print("Dataset loaded & prepared successfully.")
        try:
            HOME_MODEL, AWAY_MODEL = fine_tune_and_train()
            print("LightGBM predictive models trained successfully.")
        except Exception as e:
            print(f"LightGBM training skipped: {e}. Running on statistical engine.")


@app.get("/api/teams")
def get_teams():
    return {"teams": LA_LIGA_TEAMS}


@app.get("/api/matchdays")
def get_matchdays():
    return {"matchdays": list(range(1, 39)), "current_matchday": 1}


@app.get("/api/fixtures")
def get_fixtures_by_matchday(matchday: int = Query(1, ge=1, le=38)):
    """Fetches matchday fixtures and formats market lines in both American & Decimal formats."""
    fixtures = get_dynamic_matchday_fixtures(matchday)

    try:
        live_odds_map = fetch_draftkings_odds_dict()
    except Exception:
        live_odds_map = {}

    for f in fixtures:
        pair_key = (f["home"], f["away"])
        market_lines = live_odds_map.get(pair_key)

        dec_h, dec_d, dec_a = None, None, None
        if market_lines:
            dec_h = market_lines.get("odds_h")
            dec_d = market_lines.get("odds_d")
            dec_a = market_lines.get("odds_a")

        if not dec_h or dec_h <= 1.0:
            l_h, l_a = predict_lambdas(f["home"], f["away"], GLOBAL_DF, HOME_MODEL, AWAY_MODEL)
            sims = simulate_match_poisson(l_h, l_a)
            dec_h = round(1.0 / sims["home_win_prob"], 2) if sims["home_win_prob"] > 0 else 2.10
            dec_d = round(1.0 / sims["draw_prob"], 2) if sims["draw_prob"] > 0 else 3.40
            dec_a = round(1.0 / sims["away_win_prob"], 2) if sims["away_win_prob"] > 0 else 3.20

        f["odds_h"] = dec_h
        f["odds_d"] = dec_d
        f["odds_a"] = dec_a
        f["odds_h_american"] = decimal_to_american(dec_h)
        f["odds_d_american"] = decimal_to_american(dec_d)
        f["odds_a_american"] = decimal_to_american(dec_a)

    return fixtures


@app.post("/api/simulate", response_model=SimulationResponse)
def run_simulation(req: MatchRequest):
    """Executes Monte Carlo simulation accepting American odds inputs."""
    dec_h = american_to_decimal(req.odds_h)
    dec_d = american_to_decimal(req.odds_d)
    dec_a = american_to_decimal(req.odds_a)

    lambda_home, lambda_away = predict_lambdas(req.home_team, req.away_team, GLOBAL_DF, HOME_MODEL, AWAY_MODEL)
    sims = simulate_match_poisson(lambda_home, lambda_away)

    fair_dec_h = 1.0 / sims["home_win_prob"] if sims["home_win_prob"] > 0 else 999.0
    fair_dec_d = 1.0 / sims["draw_prob"] if sims["draw_prob"] > 0 else 999.0
    fair_dec_a = 1.0 / sims["away_win_prob"] if sims["away_win_prob"] > 0 else 999.0

    suggestions = generate_betting_insights(sims, dec_h, dec_d, dec_a)

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "sims": sims,
        "fair_odds": {
            "odds_h_american": decimal_to_american(fair_dec_h),
            "odds_d_american": decimal_to_american(fair_dec_d),
            "odds_a_american": decimal_to_american(fair_dec_a),
            "odds_h": round(fair_dec_h, 2),
            "odds_d": round(fair_dec_d, 2),
            "odds_a": round(fair_dec_a, 2),
        },
        "suggestions": suggestions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)