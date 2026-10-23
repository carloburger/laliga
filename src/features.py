# src/features.py
import pandas as pd
import numpy as np

def create_rolling_features(df: pd.DataFrame, spans: list[int] = [5, 10]) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    
    df['home_xg'] = df['home_xg'].fillna(df['FTHG'])
    df['away_xg'] = df['away_xg'].fillna(df['FTAG'])
    
    # Expand dataframe into team-centric rows
    home_stats = df[['Date', 'HomeTeam', 'home_xg', 'away_xg']].rename(
        columns={'HomeTeam': 'Team', 'home_xg': 'xG_created', 'away_xg': 'xG_conceded'}
    )
    away_stats = df[['Date', 'AwayTeam', 'away_xg', 'home_xg']].rename(
        columns={'AwayTeam': 'Team', 'away_xg': 'xG_created', 'home_xg': 'xG_conceded'}
    )
    
    team_df = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date']).reset_index(drop=True)
    
    # Compute lagged EWMA per team
    for span in spans:
        team_df[f'xg_created_ewma_{span}'] = team_df.groupby('Team')['xG_created'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
        team_df[f'xg_conceded_ewma_{span}'] = team_df.groupby('Team')['xG_conceded'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=3).mean()
        )
    
    # Merge back to Home and Away sides
    for span in spans:
        df = df.merge(
            team_df[['Date', 'Team', f'xg_created_ewma_{span}', f'xg_conceded_ewma_{span}']],
            left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left'
        ).rename(columns={
            f'xg_created_ewma_{span}': f'home_xg_created_{span}',
            f'xg_conceded_ewma_{span}': f'home_xg_conceded_{span}'
        }).drop(columns=['Team'])
        
        df = df.merge(
            team_df[['Date', 'Team', f'xg_created_ewma_{span}', f'xg_conceded_ewma_{span}']],
            left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left'
        ).rename(columns={
            f'xg_created_ewma_{span}': f'away_xg_created_{span}',
            f'xg_conceded_ewma_{span}': f'away_xg_conceded_{span}'
        }).drop(columns=['Team'])
        
    return df

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = create_rolling_features(df, spans=[5, 10])
    df = df.dropna(subset=['home_xg_created_5', 'away_xg_created_5']).reset_index(drop=True)
    return df