# src/train.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from features import prepare_dataset

# Silence Optuna logs for clean terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Load Data
DATA_PATH = Path(__file__).parent.parent / "data" / "laliga_merged_2018_2025.csv"
raw_df = pd.read_csv(DATA_PATH)
df = prepare_dataset(raw_df)

# Feature set for prediction
FEATURE_COLS = [
    'home_xg_created_5', 'home_xg_conceded_5',
    'away_xg_created_5', 'away_xg_conceded_5',
    'home_xg_created_10', 'home_xg_conceded_10',
    'away_xg_created_10', 'away_xg_conceded_10',
    'pin_prob_h', 'pin_prob_d', 'pin_prob_a'
]

# Temporal Walk-Forward Split (Train on 2018-2023, Validate on 2024+)
season_code_int = df['season_code'].astype(int)
train_mask = df['season_code'] < 2425
val_mask = df['season_code'] >= 2425

X_train, y_train_h, y_train_a = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, 'FTHG'], df.loc[train_mask, 'FTAG']
X_val, y_val_h, y_val_a = df.loc[val_mask, FEATURE_COLS], df.loc[val_mask, 'FTHG'], df.loc[val_mask, 'FTAG']

def objective(trial, is_home: bool):
    """Optuna objective function for tuning LightGBM Poisson Regressor."""
    params = {
        'objective': 'poisson',
        'metric': 'poisson',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 250),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 15, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'verbose': -1
    }
    
    y_tr = y_train_h if is_home else y_train_a
    y_v = y_val_h if is_home else y_val_a
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_tr)
    
    preds = model.predict(X_val)
    # Calculate Mean Poisson Deviance
    loss = np.mean(2 * (y_v * np.log(np.maximum(1e-5, y_v / preds)) - (y_v - preds)))
    return loss

def fine_tune_and_train():
    print("Fine-tuning Home Goals Model with Optuna...")
    study_home = optuna.create_study(direction='minimize')
    study_home.optimize(lambda t: objective(t, is_home=True), n_trials=30)
    print("Best Home Model Loss:", study_home.best_value)
    
    print("\nFine-tuning Away Goals Model with Optuna...")
    study_away = optuna.create_study(direction='minimize')
    study_away.optimize(lambda t: objective(t, is_home=False), n_trials=30)
    print("Best Away Model Loss:", study_away.best_value)
    
    # Train final models with optimal parameters
    best_home_model = lgb.LGBMRegressor(**study_home.best_params, objective='poisson', verbose=-1)
    best_home_model.fit(X_train, y_train_h)
    
    best_away_model = lgb.LGBMRegressor(**study_away.best_params, objective='poisson', verbose=-1)
    best_away_model.fit(X_train, y_train_a)
    
    return best_home_model, best_away_model

if __name__ == "__main__":
    home_model, away_model = fine_tune_and_train()
    print("\nSuccessfully trained fine-tuned LightGBM goal models!")