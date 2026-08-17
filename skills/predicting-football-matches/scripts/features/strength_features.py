#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/processed/epl_all_features.csv"
)

OUTPUT = Path(
    "data/processed/epl_strength_features.csv"
)


def build_strength_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # LEAGUE BASELINES
    #
    # All baselines are lagged so that the current match
    # never contributes to its own features.
    # --------------------------------------------------

    df["league_home_xg"] = (
        df["home_xg"]
        .shift(1)
        .expanding()
        .mean()
    )

    df["league_away_xg"] = (
        df["away_xg"]
        .shift(1)
        .expanding()
        .mean()
    )

    df["league_home_xga"] = (
        df["away_xg"]
        .shift(1)
        .expanding()
        .mean()
    )

    df["league_away_xga"] = (
        df["home_xg"]
        .shift(1)
        .expanding()
        .mean()
    )

    # --------------------------------------------------
    # ATTACK STRENGTH
    #
    # > 1.00 = above league average
    # < 1.00 = below league average
    # --------------------------------------------------

    df["home_attack_strength_5"] = (
        df["home_xg_5"]
        / df["league_home_xg"]
    )

    df["away_attack_strength_5"] = (
        df["away_xg_5"]
        / df["league_away_xg"]
    )

    df["home_attack_strength_10"] = (
        df["home_xg_10"]
        / df["league_home_xg"]
    )

    df["away_attack_strength_10"] = (
        df["away_xg_10"]
        / df["league_away_xg"]
    )

    # --------------------------------------------------
    # DEFENSIVE STRENGTH
    #
    # Lower xGA is better.
    #
    # Therefore:
    #
    # league xGA / team xGA
    #
    # > 1.00 = better defence
    # < 1.00 = weaker defence
    # --------------------------------------------------

    df["home_defence_strength_5"] = (
        df["league_home_xga"]
        / df["home_xga_5"]
    )

    df["away_defence_strength_5"] = (
        df["league_away_xga"]
        / df["away_xga_5"]
    )

    df["home_defence_strength_10"] = (
        df["league_home_xga"]
        / df["home_xga_10"]
    )

    df["away_defence_strength_10"] = (
        df["league_away_xga"]
        / df["away_xga_10"]
    )

    # --------------------------------------------------
    # RECENT xG TREND
    # --------------------------------------------------

    df["home_xg_trend"] = (
        df["home_xg_3"]
        - df["home_xg_10"]
    )

    df["away_xg_trend"] = (
        df["away_xg_3"]
        - df["away_xg_10"]
    )

    # --------------------------------------------------
    # DEFENSIVE TREND
    #
    # Negative = xGA improving
    # Positive = xGA worsening
    # --------------------------------------------------

    df["home_xga_trend"] = (
        df["home_xga_3"]
        - df["home_xga_10"]
    )

    df["away_xga_trend"] = (
        df["away_xga_3"]
        - df["away_xga_10"]
    )

    # --------------------------------------------------
    # GOALS VS xG
    #
    # Positive = scoring more goals than xG
    # Negative = scoring fewer goals than xG
    # --------------------------------------------------

    df["home_finishing_overperformance"] = (
        df["home_goal_xg_diff_5"]
    )

    df["away_finishing_overperformance"] = (
        df["away_goal_xg_diff_5"]
    )

    # --------------------------------------------------
    # MATCHUP ATTACK STRENGTH
    #
    # Team attack × opponent defence
    # --------------------------------------------------

    df["home_matchup_attack"] = (
        df["home_attack_strength_5"]
        * df["away_defence_strength_5"]
    )

    df["away_matchup_attack"] = (
        df["away_attack_strength_5"]
        * df["home_defence_strength_5"]
    )

    # --------------------------------------------------
    # MATCHUP DIFFERENCE
    # --------------------------------------------------

    df["xg_matchup_difference"] = (
        df["home_xg_diff_5"]
        - df["away_xg_diff_5"]
    )

    # --------------------------------------------------
    # HOME ADVANTAGE
    # --------------------------------------------------

    df["home_advantage_xg"] = (
        df["league_home_xg"]
        - df["league_away_xg"]
    )

    return df


if __name__ == "__main__":

    print("Loading features...")

    df = pd.read_csv(SOURCE)

    result = build_strength_features(df)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(f"Matches: {len(result)}")
    print(f"Features: {len(result.columns)}")
    print(f"Saved: {OUTPUT}")

    preview = [
        "date",
        "home_team",
        "away_team",
        "home_attack_strength_5",
        "away_attack_strength_5",
        "home_defence_strength_5",
        "away_defence_strength_5",
        "home_matchup_attack",
        "away_matchup_attack",
        "xg_matchup_difference",
    ]

    print()

    print(
        result[preview]
        .dropna()
        .head(15)
        .to_string(index=False)
    )
