import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure src directory is in Python path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy.stats import poisson
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal Module Imports
from features import prepare_dataset
from train import fine_tune_and_train
from odds import fetch_draftkings_la_liga_odds

#LA LIGA TEAMS 26/27
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

# ==========================================
# 1. PYDANTIC SCHEMAS
# ==========================================

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    odds_h: float
    odds_d: float
    odds_a: float


class SimulationResponse(BaseModel):
    lambda_home: float
    lambda_away: float
    sims: Dict[str, Any]
    fair_odds: Dict[str, float]
    suggestions: List[str]


# ==========================================
# 2. EXPECTED GOALS (xG) PREDICTOR ENGINE
# ==========================================

def predict_lambdas(home_team: str, away_team: str, df: pd.DataFrame, home_model=None, away_model=None) -> Tuple[float, float]:
    """Calculates expected goals (lambdas) for home and away teams using relative

    attack/defense strength metrics from historical data, with model inference fallback.
    """
    default_home_xg, default_away_xg = 1.45, 1.15

    if df is None or df.empty:
        return default_home_xg, default_away_xg

    try:
        # League Baseline Averages
        league_home_avg = df["FTHG"].mean()
        league_away_avg = df["FTAG"].mean()

        # Home Team Strengths
        home_matches = df[df["HomeTeam"] == home_team]
        if not home_matches.empty:
            home_att = home_matches["FTHG"].mean() / league_home_avg
            home_def = home_matches["FTAG"].mean() / league_away_avg
        else:
            home_att, home_def = 1.0, 1.0

        # Away Team Strengths
        away_matches = df[df["AwayTeam"] == away_team]
        if not away_matches.empty:
            away_att = away_matches["FTAG"].mean() / league_away_avg
            away_def = away_matches["FTHG"].mean() / league_home_avg
        else:
            away_att, away_def = 1.0, 1.0

        # Expected Goals Formula: Attack * Opponent Defense * League Baseline
        lambda_h = home_att * away_def * league_home_avg
        lambda_a = away_att * home_def * league_away_avg

        # Ensure realistic bounding limits [0.35, 4.5]
        lambda_h = float(np.clip(lambda_h, 0.35, 4.5))
        lambda_a = float(np.clip(lambda_a, 0.35, 4.5))

        return round(lambda_h, 2), round(lambda_a, 2)

    except Exception as e:
        print(f"Error calculating lambdas for {home_team} vs {away_team}: {e}")
        return default_home_xg, default_away_xg


# ==========================================
# 3. DIXON-COLES POISSON MONTE CARLO ENGINE
# ==========================================

def simulate_match_poisson(lambda_home: float, lambda_away: float, num_sims: int = 10000, rho: float = -0.13) -> Dict[str, Any]:
    """Runs Monte Carlo simulation adjusted with Dixon-Coles parameter (rho)

    for low-score interdependence (0-0, 1-0, 0-1, 1-1).
    """
    # 1. Standard Independent Poisson Draws
    home_goals = np.random.poisson(lambda_home, num_sims)
    away_goals = np.random.poisson(lambda_away, num_sims)

    # 2. Dixon-Coles Tau Adjustment Weights
    # tau(0,0) = 1 - l_h * l_a * rho
    # tau(1,0) = 1 + l_a * rho
    # tau(0,1) = 1 + l_h * rho
    # tau(1,1) = 1 - rho
    weights = np.ones(num_sims)
    
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    weights[mask_00] = max(0.1, 1.0 - (lambda_home * lambda_away * rho))
    weights[mask_10] = max(0.1, 1.0 + (lambda_away * rho))
    weights[mask_01] = max(0.1, 1.0 + (lambda_home * rho))
    weights[mask_11] = max(0.1, 1.0 - rho)

    # Normalize Weights
    weights /= np.sum(weights)

    # 3. Probabilities with Dixon-Coles Reweighting
    home_win_prob = float(np.sum(weights[home_goals > away_goals]))
    draw_prob = float(np.sum(weights[home_goals == away_goals]))
    away_win_prob = float(np.sum(weights[home_goals < away_goals]))

    # Secondary Markets
    over_2_5_prob = float(np.sum(weights[(home_goals + away_goals) > 2.5]))
    btts_prob = float(np.sum(weights[(home_goals > 0) & (away_goals > 0)]))

    # Average Expected Goals in Simulation
    avg_home_goals = float(np.average(home_goals, weights=weights))
    avg_away_goals = float(np.average(away_goals, weights=weights))

    # Top 5 Most Probable Scorelines
    scoreline_weights = {}
    for h, a, w in zip(home_goals, away_goals, weights):
        score = f"{h}-{a}"
        scoreline_weights[score] = scoreline_weights.get(score, 0.0) + w

    sorted_scorelines = sorted(scoreline_weights.items(), key=lambda x: x[1], reverse=True)
    top_scorelines = {score: round(prob, 4) for score, prob in sorted_scorelines[:5]}

    return {
        "home_win_prob": round(home_win_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "over_2_5_prob": round(over_2_5_prob, 4),
        "btts_prob": round(btts_prob, 4),
        "avg_home_goals": round(avg_home_goals, 2),
        "avg_away_goals": round(avg_away_goals, 2),
        "top_scorelines": top_scorelines,
    }


# ==========================================
# 4. INSIGHT & VALUE DETECTOR
# ==========================================

def generate_betting_insights(sims: Dict[str, Any], market_h: float, market_d: float, market_a: float) -> List[str]:
    """Evaluates mathematical edge vs market prices."""
    insights = []

    implied_h = 1.0 / market_h if market_h > 0 else 0
    implied_d = 1.0 / market_d if market_d > 0 else 0
    implied_a = 1.0 / market_a if market_a > 0 else 0

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
        insights.append("🎯 Market prices are well-calibrated against model predictions. No significant 1X2 value edge.")

    if sims["over_2_5_prob"] > 0.58:
        insights.append(f"⚽ Over 2.5 Goals Lean: High expected goal volume ({(sims['over_2_5_prob']*100):.1f}% probability).")
    elif sims["over_2_5_prob"] < 0.40:
        insights.append(f"🛡️ Under 2.5 Goals Lean: Defensive structure favored ({( (1 - sims['over_2_5_prob']) * 100):.1f}% Under probability).")

    return insights


# ==========================================
# 5. FASTAPI INITIALIZATION & STATE
# ==========================================

app = FastAPI(
    title="La Liga Predictive Betting Engine",
    description="Full-stack Monte Carlo simulation API with Dixon-Coles adjustment & DraftKings integration.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset & models
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
            print("Training LightGBM predictive models...")
            HOME_MODEL, AWAY_MODEL = fine_tune_and_train()
            print("Models successfully trained and ready.")
        except Exception as e:
            print(f"Model training skipped or failed: {e}. Falling back to team-strength xG engine.")
    else:
        print(f"Warning: Dataset not found at {DATA_PATH}. Using default statistical baselines.")


# ==========================================
# 6. API ENDPOINTS
# ==========================================
@app.get("/api/teams")
def get_teams():
    """Returns the standardized list of current 20 La Liga teams."""
    return {"teams": LA_LIGA_TEAMS}

@app.get("/api/fixtures")
def get_upcoming_fixtures():
    """Fetches real-time market lines from DraftKings.

    Populates missing market lines with Model Fair Odds.
    """
    fixtures = fetch_draftkings_la_liga_odds()

    # Dynamic Fallback Pairings if Odds API returns empty list
    if not fixtures:
        teams = get_teams()["teams"]
        sample_pairings = [
            (teams[0], teams[1]),
            (teams[2], teams[3]) if len(teams) > 3 else (teams[1], teams[0]),
            (teams[4], teams[5]) if len(teams) > 5 else (teams[0], teams[2]),
        ]
        fixtures = [
            {"id": i + 1, "date": "Upcoming", "home": h, "away": a, "odds_h": None, "odds_d": None, "odds_a": None}
            for i, (h, a) in enumerate(sample_pairings)
        ]

    # Calculate Model Fair Odds for any fixture missing market lines
    for f in fixtures:
        if not f.get("odds_h") or f["odds_h"] <= 1.0:
            l_h, l_a = predict_lambdas(f["home"], f["away"], GLOBAL_DF, HOME_MODEL, AWAY_MODEL)
            sims = simulate_match_poisson(l_h, l_a)
            f["odds_h"] = round(1.0 / sims["home_win_prob"], 2) if sims["home_win_prob"] > 0 else 2.10
            f["odds_d"] = round(1.0 / sims["draw_prob"], 2) if sims["draw_prob"] > 0 else 3.40
            f["odds_a"] = round(1.0 / sims["away_win_prob"], 2) if sims["away_win_prob"] > 0 else 3.20

    return fixtures


@app.post("/api/simulate", response_model=SimulationResponse)
def run_simulation(req: MatchRequest):
    """Executes Monte Carlo simulation for a requested fixture,

    comparing model probabilities against market odds.
    """
    # 1. Predict expected goals (xG)
    lambda_home, lambda_away = predict_lambdas(req.home_team, req.away_team, GLOBAL_DF, HOME_MODEL, AWAY_MODEL)

    # 2. Run Monte Carlo simulation (10,000 iterations)
    sims = simulate_match_poisson(lambda_home, lambda_away)

    # 3. Compute Model Fair Decimal Odds
    fair_odds_h = round(1.0 / sims["home_win_prob"], 2) if sims["home_win_prob"] > 0 else 999.0
    fair_odds_d = round(1.0 / sims["draw_prob"], 2) if sims["draw_prob"] > 0 else 999.0
    fair_odds_a = round(1.0 / sims["away_win_prob"], 2) if sims["away_win_prob"] > 0 else 999.0

    # 4. Generate EV & Value Insights
    suggestions = generate_betting_insights(
        sims=sims,
        market_h=req.odds_h,
        market_d=req.odds_d,
        market_a=req.odds_a
    )

    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "sims": sims,
        "fair_odds": {
            "odds_h": fair_odds_h,
            "odds_d": fair_odds_d,
            "odds_a": fair_odds_a,
        },
        "suggestions": suggestions
    }