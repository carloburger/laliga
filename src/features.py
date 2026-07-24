# src/features.py
import pandas as pd
import numpy as np

def create_rolling_features(df: pd.DataFrame, spans: list[int] = [5, 10]) -> pd.DataFrame:
    """
    Computes rolling Exponentially Weighted Moving Averages (EWMA) for xG created and conceded.
    Crucial: Features are shifted by 1 game so future match data NEVER leaks into the past!
    """
    df = df.sort_values("Date").reset_index(drop=True)
    
    # 1. Expand dataframe into team-centric rows
    home_stats = df[['Date', 'HomeTeam', 'home_xg', 'away_xg']].rename(
        columns={'HomeTeam': 'Team', 'home_xg': 'xG_created', 'away_xg': 'xG_conceded'}
    )
    away_stats = df[['Date', 'AwayTeam', 'away_xg', 'home_xg']].rename(
        columns={'AwayTeam': 'Team', 'away_xg': 'xG_created', 'home_xg': 'xG_conceded'}
    )
    
    team_df = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date']).reset_index(drop=True)
    
    # 2. Compute lagged EWMA per team
    for span in spans:
        team_df[f'xg_created_ewma_{span}'] = team_df.groupby('Team')['xG_created'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        team_df[f'xg_conceded_ewma_{span}'] = team_df.groupby('Team')['xG_conceded'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
    
    # 3. Merge back to Home and Away sides
    for span in spans:
        # Home team stats
        df = df.merge(
            team_df[['Date', 'Team', f'xg_created_ewma_{span}', f'xg_conceded_ewma_{span}']],
            left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left'
        ).rename(columns={
            f'xg_created_ewma_{span}': f'home_xg_created_{span}',
            f'xg_conceded_ewma_{span}': f'home_xg_conceded_{span}'
        }).drop(columns=['Team'])
        
        # Away team stats
        df = df.merge(
            team_df[['Date', 'Team', f'xg_created_ewma_{span}', f'xg_conceded_ewma_{span}']],
            left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left'
        ).rename(columns={
            f'xg_created_ewma_{span}': f'away_xg_created_{span}',
            f'xg_conceded_ewma_{span}': f'away_xg_conceded_{span}'
        }).drop(columns=['Team'])
        
    return df

def extract_market_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts margin-removed true probabilities from Pinnacle closing odds (PSH, PSD, PSA)
    to serve as a benchmark feature.
    """
    if {'PSH', 'PSD', 'PSA'}.issubset(df.columns):
        raw_h = 1 / df['PSH']
        raw_d = 1 / df['PSD']
        raw_a = 1 / df['PSA']
        margin = raw_h + raw_d + raw_a
        
        df['pin_prob_h'] = raw_h / margin
        df['pin_prob_d'] = raw_d / margin
        df['pin_prob_a'] = raw_a / margin
    return df

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = create_rolling_features(df, spans=[5, 10])
    df = extract_market_implied_probs(df)
    
    # Drop initial games without sufficient rolling history
    df = df.dropna(subset=['home_xg_created_5', 'away_xg_created_5']).reset_index(drop=True)
    return df