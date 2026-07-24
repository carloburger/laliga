import io
import os
from pathlib import Path
import ssl
import pandas as pd
import requests
import soccerdata as sd

# Disable SSL verification context for urllib on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Define project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Map team names from both sources into a clean, unified standard
TEAM_NAME_MAP = {
    # Football-Data.co.uk names -> Standard
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Betis": "Real Betis",
    "Espanol": "Espanyol",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    "Celta": "Celta Vigo",
    "Alaves": "Alaves",
    "Leganes": "Leganes",
    "Valladolid": "Real Valladolid",
    
    # Understat names -> Standard
    "Athletic Bilbao": "Athletic Club",
    "Deportivo Alaves": "Alaves",
}

def _standardize_names(series: pd.Series) -> pd.Series:
    """Helper to clean and map team names."""
    return series.replace(TEAM_NAME_MAP).str.strip()

def fetch_results_and_odds(start_year: int = 2022, end_year: int = 2024) -> pd.DataFrame:
    """
    Fetches match scores and closing bookmaker odds directly from Football-Data.co.uk CSVs.
    SP1 is the division code for La Liga Primera Division.
    """
    all_seasons = []
    
    for year in range(start_year, end_year + 1):
        # Format season code (e.g. 2022 -> '2223', 2023 -> '2324')
        season_code = f"{str(year)[-2:]}{str(year+1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/SP1.csv"
        
        try:
            print(f"[Football-Data] Fetching La Liga {year}/{year+1} from {url}...")
            # Stream CSV via requests to bypass macOS urllib SSL issues
            response = requests.get(url, verify=False, timeout=10)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            df["season_code"] = season_code
            all_seasons.append(df)
        except Exception as e:
            print(f"Skipping year {year} due to fetch error: {e}")

    if not all_seasons:
        raise ValueError("Failed to fetch any data from Football-Data.co.uk.")

    full_df = pd.concat(all_seasons, ignore_index=True)
    
    # Clean team names
    full_df["HomeTeam"] = _standardize_names(full_df["HomeTeam"])
    full_df["AwayTeam"] = _standardize_names(full_df["AwayTeam"])
    
    # Keep key score and bookmaker odds columns (Pinnacle & Bet365)
    cols_to_keep = [
        "season_code", "Date", "HomeTeam", "AwayTeam", 
        "FTHG", "FTAG", "FTR", 
        "PSH", "PSD", "PSA",   # Pinnacle Closing Odds (Home, Draw, Away)
        "B365H", "B365D", "B365A"  # Bet365 Closing Odds
    ]
    existing_cols = [c for c in cols_to_keep if c in full_df.columns]
    
    return full_df[existing_cols]

def fetch_understat_xg(seasons: list[int]) -> pd.DataFrame:
    """Fetches match-level expected goals (xG) from Understat via soccerdata."""
    print(f"[Understat] Fetching La Liga xG for seasons {seasons}...")
    
    understat = sd.Understat(leagues="ESP-La Liga", seasons=seasons)
    schedule = understat.read_schedule()
    
    # Reset multi-index to standard columns
    schedule = schedule.reset_index()
    
    # Clean team names
    schedule["home_team"] = _standardize_names(schedule["home_team"])
    schedule["away_team"] = _standardize_names(schedule["away_team"])
    
    # Format season code to match "2223", "2324", etc.
    schedule["season_code"] = schedule["season"].apply(
        lambda s: f"{str(s)[-2:]}{str(int(s)+1)[-2:]}"
    )
    
    # Keep relevant xG columns
    xg_df = schedule[[
        "season_code", "home_team", "away_team", 
        "home_xg", "away_xg"
    ]].copy()
    
    return xg_df

def get_laliga_dataset(
    start_year: int = 2022, 
    end_year: int = 2024, 
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main loader function. Merges scores, odds, and xG into a single DataFrame.
    Caches the result to a CSV file in the data/ directory.
    """
    cache_file = DATA_DIR / f"laliga_merged_{start_year}_{end_year}.csv"
    
    # 1. Return cached file if it exists and refresh isn't forced
    if cache_file.exists() and not force_refresh:
        print(f"Loading cached La Liga data from {cache_file}...")
        return pd.read_csv(cache_file)
    
    # 2. Fetch both datasets
    seasons_list = list(range(start_year, end_year + 1))
    odds_df = fetch_results_and_odds(start_year, end_year)
    xg_df = fetch_understat_xg(seasons_list)
    
    # 3. Merge on Season + Home Team + Away Team
    print("Merging results, odds, and xG stats...")
    merged_df = pd.merge(
        odds_df,
        xg_df,
        left_on=["season_code", "HomeTeam", "AwayTeam"],
        right_on=["season_code", "home_team", "away_team"],
        how="left"
    )
    
    # Drop redundant merged columns
    merged_df = merged_df.drop(columns=["home_team", "away_team"])
    
    # Sort chronologically
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], format="mixed", dayfirst=True)
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    
    # 4. Save cache to CSV
    merged_df.to_csv(cache_file, index=False)
    print(f"Successfully saved {len(merged_df)} matches to {cache_file}")
    
    return merged_df

if __name__ == "__main__":
    # Disable SSL warnings outputted by requests verify=False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Test execution: Load 2022/23 to 2024/25 seasons
    df = get_laliga_dataset(start_year=2018, end_year=2026, force_refresh=True)
    print("\n--- DATASET PREVIEW ---")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "home_xg", "away_xg", "PSH", "PSA"]].head())