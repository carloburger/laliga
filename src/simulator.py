# src/simulator.py
import numpy as np
import pandas as pd
from scipy.stats import poisson

def simulate_match(
    lambda_home: float, 
    lambda_away: float, 
    rho: float = -0.13, 
    n_sims: int = 10000
) -> dict:
    """
    Simulates `n_sims` match iterations using Poisson distributions with 
    a Dixon-Coles tau correction for low-scoring tie/win dependencies.
    """
    max_goals = 10
    goals = np.arange(max_goals + 1)
    
    # Independent Poisson grid
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    grid = np.outer(p_home, p_away)
    
    # Apply Dixon-Coles tau adjustment for low-scoring outcomes
    grid[0, 0] *= max(0, 1 - lambda_home * lambda_away * rho)
    grid[1, 0] *= max(0, 1 + lambda_away * rho)
    grid[0, 1] *= max(0, 1 + lambda_home * rho)
    grid[1, 1] *= max(0, 1 - rho)
    
    # Normalize grid probabilities to sum to 1.0
    grid /= grid.sum()
    
    # Categorical sampling across 10,000 simulations
    flat_probs = grid.flatten()
    sampled_indices = np.random.choice(len(flat_probs), size=n_sims, p=flat_probs)
    
    # Convert sampled indices back into (home_goals, away_goals)
    home_goals = sampled_indices // (max_goals + 1)
    away_goals = sampled_indices % (max_goals + 1)
    
    # Simulated Outcome Counts
    home_wins = np.sum(home_goals > away_goals)
    draws = np.sum(home_goals == away_goals)
    away_wins = np.sum(home_goals < away_goals)
    
    over_25 = np.sum((home_goals + away_goals) > 2.5)
    btts = np.sum((home_goals > 0) & (away_goals > 0))
    
    # Top 5 Most Frequent Scorelines
    score_strings = [f"{h}-{a}" for h, a in zip(home_goals, away_goals)]
    top_scores = (pd.Series(score_strings).value_counts() / n_sims).head(5).to_dict()
    
    return {
        "home_win_prob": home_wins / n_sims,
        "draw_prob": draws / n_sims,
        "away_win_prob": away_wins / n_sims,
        "over_2_5_prob": over_25 / n_sims,
        "under_2_5_prob": (n_sims - over_25) / n_sims,
        "btts_prob": btts / n_sims,
        "avg_home_goals": float(home_goals.mean()),
        "avg_away_goals": float(away_goals.mean()),
        "top_scorelines": top_scores
    }

def evaluate_simulated_bets(
    sim_results: dict, 
    odds_h: float, 
    odds_d: float, 
    odds_a: float,
    min_confidence: float = 0.45,
    min_edge: float = 0.03,
    max_edge: float = 0.15
) -> list[str]:
    """
    Suggests bets only when:
    1. The outcome has high simulated probability (Confidence).
    2. The model has a sensible edge over market implied probability (3% - 15%).
    3. Flawed outlier bets (>15% edge) are filtered out.
    """
    suggestions = []
    
    markets = [
        ("Home Win", sim_results["home_win_prob"], odds_h),
        ("Draw", sim_results["draw_prob"], odds_d),
        ("Away Win", sim_results["away_win_prob"], odds_a)
    ]
    
    for name, sim_prob, decimal_odds in markets:
        implied_prob = 1.0 / decimal_odds
        edge = sim_prob - implied_prob
        
        if sim_prob >= min_confidence and min_edge <= edge <= max_edge:
            suggestions.append(
                f"RECOMMENDED BET: {name} | Simulated: {sim_prob:.1%} vs Implied: {implied_prob:.1%} (Edge: +{edge:.1%})"
            )
            
    if not suggestions:
        suggestions.append("NO BET RECOMMENDED: Market line is sharp or model confidence is too low.")
        
    return suggestions