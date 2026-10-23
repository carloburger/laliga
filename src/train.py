# src/train.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from features import prepare_dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = Path(__file__).parent.parent / "data" / "laliga_merged_2018_2025.csv"

FEATURE_COLS = [
    'home_xg_created_5', 'home_xg_conceded_5',
    'away_xg_created_5', 'away_xg_conceded_5',
    'home_xg_created_10', 'home_xg_conceded_10',
    'away_xg_created_10', 'away_xg_conceded_10'
]

def walk_forward_cv(model_params, df, target_col):
    if df is None or df.empty:
        raise ValueError("DataFrame passed to walk_forward_cv is None or empty.")

    seasons = sorted(df['season_code'].unique())
    train_seasons = seasons[:3] 
    val_seasons = seasons[3:]   
    
    total_loss = 0
    folds = 0
    current_train_seasons = list(train_seasons)
    
    for val_season in val_seasons:
        train_mask = df['season_code'].isin(current_train_seasons)
        val_mask = df['season_code'] == val_season
        
        X_tr, y_tr = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, target_col]
        X_v, y_v = df.loc[val_mask, FEATURE_COLS], df.loc[val_mask, target_col]
        
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_v)
        loss = np.mean(2 * (y_v * np.log(np.maximum(1e-5, y_v / preds)) - (y_v - preds)))
        
        total_loss += loss
        folds += 1
        current_train_seasons.append(val_season)
        
    return total_loss / folds

def objective(trial, df, is_home: bool):
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
    
    target = 'FTHG' if is_home else 'FTAG'
    return walk_forward_cv(params, df, target)

def fine_tune_and_train(df: pd.DataFrame = None):
    # Ensure df is loaded if None or invalid
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raw_df = pd.read_csv(DATA_PATH)
        df = prepare_dataset(raw_df)

    print("Fine-tuning Home Goals Model with Walk-Forward CV...")
    study_home = optuna.create_study(direction='minimize')
    study_home.optimize(lambda t: objective(t, df, is_home=True), n_trials=30)
    print(f"Best Home Model Poisson Loss: {study_home.best_value:.4f}")
    
    print("\nFine-tuning Away Goals Model with Walk-Forward CV...")
    study_away = optuna.create_study(direction='minimize')
    study_away.optimize(lambda t: objective(t, df, is_home=False), n_trials=30)
    print(f"Best Away Model Poisson Loss: {study_away.best_value:.4f}")
    
    final_home_model = lgb.LGBMRegressor(**study_home.best_params, objective='poisson', verbose=-1)
    final_home_model.fit(df[FEATURE_COLS], df['FTHG'])
    
    final_away_model = lgb.LGBMRegressor(**study_away.best_params, objective='poisson', verbose=-1)
    final_away_model.fit(df[FEATURE_COLS], df['FTAG'])
    
    return final_home_model, final_away_model

if __name__ == "__main__":
    raw_df = pd.read_csv(DATA_PATH)
    dataset = prepare_dataset(raw_df)
    home_model, away_model = fine_tune_and_train(dataset)