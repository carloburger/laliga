import penaltyblog as pb
from data_loader import load_laliga_data
from team_mappings import standardize_team_names

def run_laliga_model():
    # 1. Fetch multi-season La Liga data (2023-2025)
    df = load_laliga_data(start_year=2023, end_year=2025)
    df = standardize_team_names(df)

    # 2. Fit Dixon-Coles Model
    print("\nFitting Dixon-Coles model on La Liga matches...")
    dc_model = pb.models.DixonColes(
        df["FTHG"], 
        df["FTAG"], 
        df["HomeTeam"], 
        df["AwayTeam"]
    )
    dc_model.fit()

    # 3. Predict an upcoming El Clásico or targeted matchup
    home_team = "Real Madrid"
    away_team = "Barcelona"

    prediction = dc_model.predict(home_team, away_team)

    print(f"\n==========================================")
    print(f" LA LIGA MODEL: {home_team} vs {away_team}")
    print(f"==========================================")
    print(f"Real Madrid Win:        {prediction.home_win:.2%}")
    print(f"Draw:                   {prediction.draw:.2%}")
    print(f"Barcelona Win:          {prediction.away_win:.2%}")
    print(f"------------------------------------------")
    print(f"Over 2.5 Goals:         {prediction.total_goals_over(2.5):.2%}")
    print(f"Under 2.5 Goals:        {1 - prediction.total_goals_over(2.5):.2%}")
    print(f"Both Teams To Score:    {prediction.both_teams_to_score:.2%}")
    print(f"==========================================\n")

if __name__ == "__main__":
    run_laliga_model()