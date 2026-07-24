# src/predict_match.py
import argparse
import sys
from pathlib import Path
import pandas as pd



from features import prepare_dataset
from train import fine_tune_and_train
from simulator import simulate_match, evaluate_simulated_bets


def load_dataset_and_models():
    """Loads historical data and returns trained models and processed DataFrame."""
    data_path = Path(__file__).parent.parent / "data" / "laliga_merged_2018_2025.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = prepare_dataset(pd.read_csv(data_path))
    print("Training LightGBM Poisson models (Home & Away)...")
    home_model, away_model = fine_tune_and_train()
    return df, home_model, away_model


def get_available_teams(df: pd.DataFrame) -> list[str]:
    """Returns sorted unique team names present in the dataset."""
    teams = sorted(list(set(df['HomeTeam'].dropna().unique())))
    return teams


def generate_match_report(
    home_team: str,
    away_team: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
    df: pd.DataFrame,
    home_model,
    away_model
):
    """Executes feature extraction, predictions, and 10,000 Monte Carlo simulations."""
    # Validate team existence in dataset
    if home_team not in df['HomeTeam'].values or away_team not in df['AwayTeam'].values:
        print(f"\n Error: One or both teams ('{home_team}', '{away_team}') non-existent in historical data.")
        return

    # Get latest rolling features
    home_stats = df[df['HomeTeam'] == home_team].iloc[-1]
    away_stats = df[df['AwayTeam'] == away_team].iloc[-1]

    features = pd.DataFrame([{
        'home_xg_created_5': home_stats['home_xg_created_5'],
        'home_xg_conceded_5': home_stats['home_xg_conceded_5'],
        'away_xg_created_5': away_stats['away_xg_created_5'],
        'away_xg_conceded_5': away_stats['away_xg_conceded_5'],
        'home_xg_created_10': home_stats['home_xg_created_10'],
        'home_xg_conceded_10': home_stats['home_xg_conceded_10'],
        'away_xg_created_10': away_stats['away_xg_created_10'],
        'away_xg_conceded_10': away_stats['away_xg_conceded_10'],
        'pin_prob_h': 1.0 / odds_h,
        'pin_prob_d': 1.0 / odds_d,
        'pin_prob_a': 1.0 / odds_a
    }])

    # Predict goal expectations (λ and μ)
    lambda_h = float(home_model.predict(features)[0])
    lambda_a = float(away_model.predict(features)[0])

    # Run 10,000 simulations
    sims = simulate_match(lambda_h, lambda_a, n_sims=10000)
    suggestions = evaluate_simulated_bets(sims, odds_h, odds_d, odds_a)

    # Display Report
    print("\n=======================================================")
    print(f" SIMULATION MATCH REPORT: {home_team} vs {away_team}")
    print("=======================================================")
    print(f" Predicted xG:           {home_team} ({lambda_h:.2f}) - ({lambda_a:.2f}) {away_team}")
    print(f" Simulated Goal Average: {home_team} ({sims['avg_home_goals']:.2f}) - ({sims['avg_away_goals']:.2f}) {away_team}")
    print("-------------------------------------------------------")
    print(f" Home Win ({home_team}):  {sims['home_win_prob']:.1%}  (Market Implied: {1/odds_h:.1%})")
    print(f" Draw:                   {sims['draw_prob']:.1%}  (Market Implied: {1/odds_d:.1%})")
    print(f" Away Win ({away_team}):  {sims['away_win_prob']:.1%}  (Market Implied: {1/odds_a:.1%})")
    print("-------------------------------------------------------")
    print(f" Over 2.5 Goals:         {sims['over_2_5_prob']:.1%}")
    print(f" Both Teams To Score:    {sims['btts_prob']:.1%}")
    print("-------------------------------------------------------")
    print(" Top 5 Most Likely Exact Scorelines:")
    for score, pct in sims['top_scorelines'].items():
        print(f"   • {score}: {pct:.1%}")
    print("-------------------------------------------------------")
    print(" BETTING RECOMMENDATIONS:")
    for sug in suggestions:
        print(f"   {sug}")
    print("=======================================================\n")


def interactive_prompt(df: pd.DataFrame, home_model, away_model):
    """Interactive CLI menu to select teams and odds."""
    teams = get_available_teams(df)

    print("\n--- LA LIGA SIMULATION ENGINE ---")
    print("Select Home Team:")
    for idx, team in enumerate(teams, 1):
        print(f"  [{idx}] {team}")

    home_idx = int(input("\nEnter Home Team number: ")) - 1
    home_team = teams[home_idx]

    print(f"\nSelect Away Team (excluding {home_team}):")
    away_teams = [t for t in teams if t != home_team]
    for idx, team in enumerate(away_teams, 1):
        print(f"  [{idx}] {team}")

    away_idx = int(input("\nEnter Away Team number: ")) - 1
    away_team = away_teams[away_idx]

    print(f"\nEnter Decimal Odds for {home_team} vs {away_team}:")
    odds_h = float(input(f"  • {home_team} Win Odds (e.g. 2.10): "))
    odds_d = float(input("  • Draw Odds (e.g. 3.40): "))
    odds_a = float(input(f"  • {away_team} Win Odds (e.g. 3.20): "))

    generate_match_report(home_team, away_team, odds_h, odds_d, odds_a, df, home_model, away_model)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Run La Liga Monte Carlo Match Simulation")
    parser.add_argument("--home", type=str, help="Home Team Name")
    parser.add_argument("--away", type=str, help="Away Team Name")
    parser.add_argument("--odds", type=float, nargs=3, metavar=('ODDS_H', 'ODDS_D', 'ODDS_A'),
                        help="Decimal odds for Home, Draw, Away (e.g., 2.10 3.40 3.20)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive match selector")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df, home_model, away_model = load_dataset_and_models()

    if args.home and args.away and args.odds:
        # CLI Flag Mode
        generate_match_report(
            home_team=args.home,
            away_team=args.away,
            odds_h=args.odds[0],
            odds_d=args.odds[1],
            odds_a=args.odds[2],
            df=df,
            home_model=home_model,
            away_model=away_model
        )
    else:
        # Default to Interactive Mode
        interactive_prompt(df, home_model, away_model)