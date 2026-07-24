# src/evaluate.py
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pathlib import Path

from features import prepare_dataset
from train import fine_tune_and_train

def poisson_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
    """Calculates a (max_goals+1) x (max_goals+1) grid of exact score probabilities."""
    goals = np.arange(0, max_goals + 1)
    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)
    return np.outer(p_home, p_away)

def match_probabilities(lambda_home: float, lambda_away: float) -> dict:
    """Aggregates score matrix into market probabilities (1X2, Over/Under 2.5, BTTS)."""
    grid = poisson_score_matrix(lambda_home, lambda_away)
    
    # 1X2 Probabilities
    prob_home = float(np.sum(np.tril(grid, -1)))  # i > j
    prob_draw = float(np.sum(np.diag(grid)))       # i == j
    prob_away = float(np.sum(np.triu(grid, 1)))   # i < j
    
    # Over / Under 2.5 Goals
    goals = np.arange(grid.shape[0])
    grid_i, grid_j = np.meshgrid(goals, goals, indexing='ij')
    prob_over_25 = float(np.sum(grid[grid_i + grid_j > 2.5]))
    
    # Both Teams To Score (BTTS)
    prob_btts = float(np.sum(grid[1:, 1:]))
    
    return {
        'model_prob_h': prob_home,
        'model_prob_d': prob_draw,
        'model_prob_a': prob_away,
        'model_prob_over25': prob_over_25,
        'model_prob_under25': 1.0 - prob_over_25,
        'model_prob_btts': prob_btts
    }

def evaluate_out_of_sample():
    # 1. Load Data and Train Models
    DATA_PATH = Path(__file__).parent.parent / "data" / "laliga_merged_2018_2025.csv"
    df = prepare_dataset(pd.read_csv(DATA_PATH))
    
    home_model, away_model = fine_tune_and_train()
    
    # 2. Filter Out-of-Sample Validation Set (2024+ seasons)
    val_df = df[df['season_code'].astype(int) >= 2425].copy().reset_index(drop=True)
    
    FEATURE_COLS = [
        'home_xg_created_5', 'home_xg_conceded_5',
        'away_xg_created_5', 'away_xg_conceded_5',
        'home_xg_created_10', 'home_xg_conceded_10',
        'away_xg_created_10', 'away_xg_conceded_10',
        'pin_prob_h', 'pin_prob_d', 'pin_prob_a'
    ]
    
    # 3. Predict Expected Goals (λ and μ)
    val_df['pred_lambda'] = home_model.predict(val_df[FEATURE_COLS])
    val_df['pred_mu'] = away_model.predict(val_df[FEATURE_COLS])
    
    # 4. Generate Market Probabilities
    probs_list = [
        match_probabilities(row['pred_lambda'], row['pred_mu']) 
        for _, row in val_df.iterrows()
    ]
    probs_df = pd.DataFrame(probs_list)
    val_df = pd.concat([val_df, probs_df], axis=1)
    
    # 5. Calculate +EV Bets against Pinnacle Closing Odds (PSH, PSD, PSA)
    val_df['ev_h'] = (val_df['model_prob_h'] * val_df['PSH']) - 1
    val_df['ev_d'] = (val_df['model_prob_d'] * val_df['PSD']) - 1
    val_df['ev_a'] = (val_df['model_prob_a'] * val_df['PSA']) - 1
    
    # 6. Model Evaluation Metrics (Brier Score)
    # Convert actual results (FTR) to one-hot encoding
    actual_h = (val_df['FTR'] == 'H').astype(int)
    actual_d = (val_df['FTR'] == 'D').astype(int)
    actual_a = (val_df['FTR'] == 'A').astype(int)
    
    brier_score = np.mean(
        (val_df['model_prob_h'] - actual_h)**2 + 
        (val_df['model_prob_d'] - actual_d)**2 + 
        (val_df['model_prob_a'] - actual_a)**2
    )
    
    print("\n==========================================")
    print(" OUT-OF-SAMPLE EVALUATION RESULTS (2024-2026)")
    print("==========================================")
    print(f" Matches Evaluated: {len(val_df)}")
    print(f" Model Brier Score: {brier_score:.4f} (Lower is better)")
    print(f" Home +EV Opportunities (>3% EV): {(val_df['ev_h'] > 0.03).sum()}")
    print(f" Draw +EV Opportunities (>3% EV): {(val_df['ev_d'] > 0.03).sum()}")
    print(f" Away +EV Opportunities (>3% EV): {(val_df['ev_a'] > 0.03).sum()}")
    print("==========================================\n")
    
    # Preview top 5 +EV Home bets
    top_ev_h = val_df[val_df['ev_h'] > 0.03][
        ['Date', 'HomeTeam', 'AwayTeam', 'PSH', 'model_prob_h', 'pin_prob_h', 'ev_h']
    ].head()
    print("Top +EV Home Bets Preview:")
    print(top_ev_h.to_string(index=False))

if __name__ == "__main__":
    evaluate_out_of_sample()