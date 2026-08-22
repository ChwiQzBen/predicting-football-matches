#!/usr/bin/env python3
"""Match prediction script using ensemble predictions and value/EV analysis."""

import argparse
import sys
from pathlib import Path
from math import exp, factorial

import pandas as pd
import numpy as np


# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


# ============================================================
# POISSON FUNCTIONS (from match_model.py)
# ============================================================

def poisson_pmf(k, lam):
    """Poisson probability mass function."""
    return exp(-lam) * lam ** k / factorial(k)


def score_matrix(home_xg, away_xg, max_goals=10):
    """Matrix of all possible score combinations."""
    return {
        (i, j): poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    }


def margin_probs(home_xg, away_xg, max_goals=10):
    """P(home_goals - away_goals == m)."""
    out = {}
    for (i, j), p in score_matrix(home_xg, away_xg, max_goals).items():
        out[i - j] = out.get(i - j, 0.0) + p
    return out


def outcome_probs(home_xg, away_xg, max_goals=10):
    """P(home_win), P(draw), P(away_win)."""
    m = margin_probs(home_xg, away_xg, max_goals)
    home = sum(p for k, p in m.items() if k > 0)
    draw = m.get(0, 0.0)
    away = sum(p for k, p in m.items() if k < 0)
    return home, draw, away


# ============================================================
# VALUE/EV FUNCTIONS
# ============================================================

def no_vig(odds):
    """Convert decimal odds to fair probabilities.
    
    odds: dict with keys "home", "draw", "away"
    Returns: (fair_probs dict, vigorish/margin)
    """
    raw = {k: 1.0 / v for k, v in odds.items()}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}, s - 1.0


def ev(prob, decimal_odds):
    """Expected value per 1 unit staked. Positive => +EV."""
    return prob * decimal_odds - 1.0


def kelly(prob, decimal_odds, fraction=0.25):
    """Fractional-Kelly stake (share of bankroll). 0 if no edge."""
    b = decimal_odds - 1.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f) * fraction


# ============================================================
# DATA LOADING
# ============================================================

def load_ensemble_predictions():
    """Load the latest ensemble predictions."""
    data_dir = SCRIPT_DIR / "data" / "processed"
    ensemble_path = data_dir / "ensemble_predictions.csv"
    
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Ensemble predictions not found: {ensemble_path}")
    
    return pd.read_csv(ensemble_path)


def get_recent_xg_averages(team_name, df, n_recent=5):
    """Get average XG for a team from recent matches.
    
    Returns: (avg_xg_for, avg_xg_against)
    """
    home_matches = df[
        (df["home_team"].str.lower() == team_name.lower())
    ].tail(n_recent)
    
    away_matches = df[
        (df["away_team"].str.lower() == team_name.lower())
    ].tail(n_recent)
    
    if len(home_matches) == 0 and len(away_matches) == 0:
        raise ValueError(f"Team '{team_name}' not found in data")
    
    # Combine home and away
    all_home_xg = list(home_matches["home_xg"].values)
    all_away_xg = list(home_matches["away_xg"].values)
    
    all_home_xg += list(away_matches["away_xg"].values)
    all_away_xg += list(away_matches["home_xg"].values)
    
    avg_xg_for = np.mean(all_home_xg) if all_home_xg else 0.0
    avg_xg_against = np.mean(all_away_xg) if all_away_xg else 0.0
    
    return avg_xg_for, avg_xg_against


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_match(home_team, away_team, odds_home, odds_draw, odds_away):
    """Predict a match and show value analysis."""
    
    # Load data
    df = load_ensemble_predictions()
    
    # Get XG estimates from recent matches
    home_xg_for, home_xg_against = get_recent_xg_averages(home_team, df)
    away_xg_for, away_xg_against = get_recent_xg_averages(away_team, df)
    
    # Adjust: home team XG for vs away team defensive XG against
    # and away team XG for vs home team defensive XG against
    # Simple average or use the team's attacking XG and opponent's defending
    home_xg = (home_xg_for + away_xg_against) / 2
    away_xg = (away_xg_for + home_xg_against) / 2
    
    # Ensure reasonable bounds
    home_xg = max(0.3, min(home_xg, 3.0))
    away_xg = max(0.3, min(away_xg, 3.0))
    
    # Calculate Poisson probabilities
    home_prob, draw_prob, away_prob = outcome_probs(home_xg, away_xg)
    
    # Calculate fair odds and vigorish
    odds = {"home": odds_home, "draw": odds_draw, "away": odds_away}
    fair_probs, margin = no_vig(odds)
    
    # Calculate EV for each outcome
    home_ev = ev(home_prob, odds_home)
    draw_ev = ev(draw_prob, odds_draw)
    away_ev = ev(away_prob, odds_away)
    
    # Calculate Kelly
    home_kelly = kelly(home_prob, odds_home)
    draw_kelly = kelly(draw_prob, odds_draw)
    away_kelly = kelly(away_prob, odds_away)
    
    # Print results
    print("\n" + "="*70)
    print(f"MATCH PREDICTION: {home_team} vs {away_team}")
    print("="*70)
    
    print(f"\nESTIMATED XG:")
    print(f"  {home_team:20s}: {home_xg:6.2f}")
    print(f"  {away_team:20s}: {away_xg:6.2f}")
    
    print(f"\nPOISSON PROBABILITIES (Model):")
    print(f"  {home_team} Win : {home_prob:6.1%}")
    print(f"  Draw        : {draw_prob:6.1%}")
    print(f"  {away_team} Win : {away_prob:6.1%}")
    
    print(f"\nMARKET ODDS & FAIR PROBABILITY:")
    print(f"  {home_team} Win : {odds_home:5.2f} (Fair: {fair_probs['home']:6.1%})")
    print(f"  Draw        : {odds_draw:5.2f} (Fair: {fair_probs['draw']:6.1%})")
    print(f"  {away_team} Win : {odds_away:5.2f} (Fair: {fair_probs['away']:6.1%})")
    print(f"  Vigorish    : {margin:6.1%}")
    
    print(f"\nEXPECTED VALUE (per unit):")
    print(f"  {home_team} Win : {home_ev:+6.3f}")
    print(f"  Draw        : {draw_ev:+6.3f}")
    print(f"  {away_team} Win : {away_ev:+6.3f}")
    
    print(f"\nFRACTIONAL KELLY (25% of full Kelly):")
    print(f"  {home_team} Win : {home_kelly:6.1%}")
    print(f"  Draw        : {draw_kelly:6.1%}")
    print(f"  {away_team} Win : {away_kelly:6.1%}")
    
    # Best bet recommendation
    print(f"\nRECOMMENDATION:")
    outcomes = {
        f"{home_team} Win": (home_ev, home_kelly),
        "Draw": (draw_ev, draw_kelly),
        f"{away_team} Win": (away_ev, away_kelly),
    }
    
    best_outcome = max(outcomes.items(), key=lambda x: x[1][0])
    if best_outcome[1][0] > 0:
        print(f"  BEST BET: {best_outcome[0]}")
        print(f"  EV: {best_outcome[1][0]:+.3f} per unit")
        print(f"  Suggested Kelly: {best_outcome[1][1]:.1%}")
    else:
        print(f"  NO +EV BETS FOUND")
        print(f"  Best option: {best_outcome[0]} with EV {best_outcome[1][0]:+.3f}")
    
    print("\n" + "="*70 + "\n")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict a match using Poisson model and value analysis"
    )
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--odds-home", type=float, required=True, help="Decimal odds for home win")
    parser.add_argument("--odds-draw", type=float, required=True, help="Decimal odds for draw")
    parser.add_argument("--odds-away", type=float, required=True, help="Decimal odds for away win")
    
    args = parser.parse_args()
    
    try:
        predict_match(
            args.home,
            args.away,
            args.odds_home,
            args.odds_draw,
            args.odds_away,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
