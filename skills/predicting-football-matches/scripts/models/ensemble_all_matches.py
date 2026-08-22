#!/usr/bin/env python3

"""
Run ensemble predictions on all 1,900 matches.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

# Load the full feature dataset
features_path = DATA_DIR / "epl_all_features.csv"
elo_path = DATA_DIR / "elo_predictions.csv"
output_path = DATA_DIR / "ensemble_predictions_all.csv"

print("Loading full feature dataset...")
features = pd.read_csv(features_path)
print(f"Features shape: {features.shape}")

print("Loading Elo predictions...")
elo = pd.read_csv(elo_path)
print(f"Elo shape: {elo.shape}")

# Merge features with Elo on date and teams
print("Merging datasets...")
merged = features.merge(
    elo[["date", "home_team", "away_team", "home_win_prob", "draw_prob", "away_win_prob"]],
    on=["date", "home_team", "away_team"],
    how="left",
    suffixes=("", "_elo")
)

print(f"Merged shape: {merged.shape}")
print(f"Matches with Elo: {merged['home_win_prob_elo'].notna().sum()}")

# Drop rows without Elo
merged = merged.dropna(subset=["home_win_prob_elo", "draw_prob_elo", "away_win_prob_elo"])

# Use calibrated probabilities (from the calibrated model)
# If we don't have calibrated probabilities for all matches, use the raw xG model probabilities
if "cal_home_win_prob" in merged.columns:
    home_col = "cal_home_win_prob"
    draw_col = "cal_draw_prob"
    away_col = "cal_away_win_prob"
else:
    # Use xG model probabilities from Poisson
    home_col = "home_xg_prob" if "home_xg_prob" in merged.columns else "home_win_prob"
    draw_col = "draw_prob" if "draw_prob" in merged.columns else "draw_prob" 
    away_col = "away_xg_prob" if "away_xg_prob" in merged.columns else "away_win_prob"
    
    # If we don't have probabilities, calculate from xG using Poisson
    if home_col not in merged.columns:
        print("Calculating Poisson probabilities from xG...")
        # Simple Poisson approximation
        # This is a simplification - in practice, use the full Poisson model
        total_xg = merged["home_xg"] + merged["away_xg"]
        merged["home_win_prob"] = 1 / (1 + np.exp(-(merged["home_xg"] - merged["away_xg"]) * 0.8))
        merged["away_win_prob"] = 1 / (1 + np.exp((merged["home_xg"] - merged["away_xg"]) * 0.8))
        merged["draw_prob"] = 0.25 - 0.15 * np.abs(merged["home_xg"] - merged["away_xg"])
        merged["draw_prob"] = np.clip(merged["draw_prob"], 0.10, 0.35)
        
        total = merged["home_win_prob"] + merged["draw_prob"] + merged["away_win_prob"]
        merged["home_win_prob"] /= total
        merged["draw_prob"] /= total
        merged["away_win_prob"] /= total
        
        home_col = "home_win_prob"
        draw_col = "draw_prob"
        away_col = "away_win_prob"

# Now create ensemble
BASE_WEIGHT = 0.70
ELO_WEIGHT = 0.30
DRAW_FLOOR = 0.10
DRAW_CEILING = 0.38
DRAW_BASE_WEIGHT = 0.80
DRAW_ELO_WEIGHT = 0.20

base_home = merged[home_col].fillna(0.33)
base_draw = merged[draw_col].fillna(0.33)
base_away = merged[away_col].fillna(0.33)

elo_home = merged["home_win_prob_elo"].fillna(0.33)
elo_draw = merged["draw_prob_elo"].fillna(0.33)
elo_away = merged["away_win_prob_elo"].fillna(0.33)

# Weighted average for non-draw probabilities
raw_home = BASE_WEIGHT * base_home + ELO_WEIGHT * elo_home
raw_away = BASE_WEIGHT * base_away + ELO_WEIGHT * elo_away

# Draw model
draw_signal = DRAW_BASE_WEIGHT * base_draw + DRAW_ELO_WEIGHT * elo_draw
base_gap = np.abs(base_home - base_away)
elo_gap = np.abs(elo_home - elo_away)
average_gap = (base_gap + elo_gap) / 2.0
uncertainty_bonus = 0.08 * np.maximum(0.0, 1.0 - average_gap / 0.30)
draw = draw_signal + uncertainty_bonus
draw = np.clip(draw, DRAW_FLOOR, DRAW_CEILING)

# Normalize
non_draw_total = raw_home + raw_away
non_draw_total = np.maximum(non_draw_total, 0.001)
available_mass = 1.0 - draw

home = raw_home / non_draw_total * available_mass
away = raw_away / non_draw_total * available_mass

merged["ensemble_home_win_prob"] = home * 100
merged["ensemble_draw_prob"] = draw * 100
merged["ensemble_away_win_prob"] = away * 100

# Add predictions
probs = np.column_stack([home, draw, away])
winner_idx = probs.argmax(axis=1)
merged["ensemble_prediction"] = np.where(winner_idx == 0, "HOME", 
                                        np.where(winner_idx == 1, "DRAW", "AWAY"))
merged["ensemble_probability"] = probs.max(axis=1) * 100

# Save
output_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
print(f"Total matches: {len(merged)}")
print(f"Prediction distribution:")
print(merged["ensemble_prediction"].value_counts())
print(f"\nAverage probabilities:")
print(f"HOME: {home.mean() * 100:.2f}%")
print(f"DRAW: {draw.mean() * 100:.2f}%")
print(f"AWAY: {away.mean() * 100:.2f}%")
print("\n✅ Ensemble predictions for all matches complete!")
