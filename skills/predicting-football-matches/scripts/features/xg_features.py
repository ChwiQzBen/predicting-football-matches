
#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROLLING_WINDOWS = (5, 10)


def _team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert home/away match records into one row per team per match.

    Every team's record contains only information that was available
    from matches already played.
    """

    home = pd.DataFrame({
        "date": df["date"],
        "match_id": df["match_id"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "venue": "home",
        "goals_for": df["home_goals"],
        "goals_against": df["away_goals"],
        "xg_for": df["home_xg"],
        "xg_against": df["away_xg"],
    })

    away = pd.DataFrame({
        "date": df["date"],
        "match_id": df["match_id"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "venue": "away",
        "goals_for": df["away_goals"],
        "goals_against": df["home_goals"],
        "xg_for": df["away_xg"],
        "xg_against": df["home_xg"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True,
    )

    history = history.sort_values(
        ["team", "date", "match_id"]
    ).reset_index(drop=True)

    return history


def _rolling_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling features using ONLY previous matches.

    shift(1) is critical: the current match can never influence
    the features used to predict itself.
    """

    history = history.copy()

    grouped = history.groupby("team", group_keys=False)

    for window in ROLLING_WINDOWS:
        history[f"xg_{window}"] = grouped["xg_for"].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"xga_{window}"] = grouped["xg_against"].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"goals_{window}"] = grouped["goals_for"].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"goals_against_{window}"] = grouped[
            "goals_against"
        ].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"xg_diff_{window}"] = (
            history[f"xg_{window}"]
            - history[f"xga_{window}"]
        )

        history[f"form_{window}"] = grouped["goals_for"].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"xg_std_{window}"] = grouped["xg_for"].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=2,
            ).std()
        )

    return history


def _home_away_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate venue-specific rolling xG features.
    """

    history = history.copy()

    venue_group = history.groupby(
        ["team", "venue"],
        group_keys=False,
    )

    for window in ROLLING_WINDOWS:
        history[f"venue_xg_{window}"] = venue_group[
            "xg_for"
        ].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

        history[f"venue_xga_{window}"] = venue_group[
            "xg_against"
        ].transform(
            lambda s: s.shift(1).rolling(
                window,
                min_periods=1,
            ).mean()
        )

    return history


def _rest_days(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()

    previous_date = history.groupby("team")["date"].shift(1)

    history["rest_days"] = (
        history["date"] - previous_date
    ).dt.total_seconds() / 86400.0

    return history


def build_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe team-level features.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    history = _team_history(df)

    history = _rolling_features(history)

    history = _home_away_features(history)

    history = _rest_days(history)

    return history


def build_match_features(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine home-team and away-team historical features
    into one prediction-ready row per match.
    """

    matches = matches.copy()

    matches["date"] = pd.to_datetime(matches["date"])

    history = build_team_features(matches)

    home = history[
        history["venue"] == "home"
    ].copy()

    away = history[
        history["venue"] == "away"
    ].copy()

    home_cols = [
        "match_id",
        "date",
        "team",
        "xg_5",
        "xg_10",
        "xga_5",
        "xga_10",
        "goals_5",
        "goals_10",
        "goals_against_5",
        "goals_against_10",
        "xg_diff_5",
        "xg_diff_10",
        "xg_std_5",
        "xg_std_10",
        "venue_xg_5",
        "venue_xg_10",
        "venue_xga_5",
        "venue_xga_10",
        "rest_days",
    ]

    away_cols = home_cols.copy()

    home = home[home_cols].rename(
        columns={
            c: f"home_{c}"
            for c in home_cols
            if c not in ("match_id", "date")
        }
    )

    away = away[away_cols].rename(
        columns={
            c: f"away_{c}"
            for c in away_cols
            if c not in ("match_id", "date")
        }
    )

    features = matches.merge(
        home,
        on=["match_id", "date"],
        how="left",
    )

    features = features.merge(
        away,
        on=["match_id", "date"],
        how="left",
    )

    return features


def save_features(
    features: pd.DataFrame,
    output_file: str | Path,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(
        output_file,
        index=False,
    )


if __name__ == "__main__":
    source = Path(
        "data/processed/epl_2025_matches.csv"
    )

    output = Path(
        "data/processed/epl_2025_features.csv"
    )

    matches = pd.read_csv(source)

    features = build_match_features(matches)

    save_features(
        features,
        output,
    )

    print(f"Matches: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(f"Saved: {output}")
    print()

    # Preview key features (using correct column names)
    preview_cols = [
        "date",
        "home_team_x",
        "away_team_x",
        "home_xg_5",
        "home_xga_5",
        "away_xg_5",
        "away_xga_5",
        "home_xg_diff_5",
        "away_xg_diff_5",
    ]
    
    print(features[preview_cols].head(15).to_string(index=False))
