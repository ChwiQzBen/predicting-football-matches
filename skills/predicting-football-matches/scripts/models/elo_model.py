#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SOURCE = DATA_DIR / "processed" / "epl_strength_features.csv"

OUTPUT_PATH = (
    DATA_DIR
    / "processed"
    / "elo_predictions.csv"
)


# ============================================================
# ELO PARAMETERS
# ============================================================

INITIAL_ELO = 1500.0

# Home advantage is applied only when calculating the
# pre-match expected result.
HOME_ADVANTAGE = 65.0

K_FACTOR = 20.0


# ============================================================
# DRAW PARAMETERS
# ============================================================

# Maximum draw probability for a very evenly matched game.
MAX_DRAW_PROB = 0.30

# Minimum draw probability for a very large Elo mismatch.
MIN_DRAW_PROB = 0.20

# Controls how quickly draw probability falls as Elo gap grows.
DRAW_GAP_SCALE = 350.0


# ============================================================
# INPUT
# ============================================================

def find_input_file() -> Path:
    """
    Use the same strength-feature dataset as the xG model.

    This avoids depending on a separate matches.csv file and
    guarantees that Elo is working from the same chronological
    dataset as the rest of the modelling pipeline.
    """

    if SOURCE.exists():
        return SOURCE

    raise FileNotFoundError(
        "\nCould not find the strength feature dataset.\n\n"
        f"Expected:\n"
        f"  {SOURCE}\n\n"
        "Run the strength feature generation step first."
    )


# ============================================================
# PREPARATION
# ============================================================

def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    required = [
        "date",
        "match_id",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["home_xg"] = pd.to_numeric(
        df["home_xg"],
        errors="coerce",
    )

    df["away_xg"] = pd.to_numeric(
        df["away_xg"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "date",
            "match_id",
            "home_team",
            "away_team",
            "home_xg",
            "away_xg",
        ]
    ).copy()

    df = df.sort_values(
        [
            "date",
            "match_id",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# ELO
# ============================================================

def elo_expected(
    rating_a: float,
    rating_b: float,
) -> float:
    """
    Expected probability of team A beating team B.
    """

    return 1.0 / (
        1.0
        + 10.0 ** (
            (rating_b - rating_a) / 400.0
        )
    )


def goal_margin_multiplier(
    goal_difference: float,
) -> float:
    """
    Conservative goal-margin multiplier.

    Larger wins receive larger Elo updates but the multiplier
    is capped to prevent extreme scorelines from dominating.
    """

    margin = abs(
        int(goal_difference)
    )

    if margin <= 1:
        return 1.00

    if margin == 2:
        return 1.50

    if margin == 3:
        return 1.75

    return 1.90


# ============================================================
# DRAW MODEL
# ============================================================

def draw_probability(
    adjusted_elo_gap: float,
) -> float:
    """
    Estimate draw probability from matchup closeness.

    Very evenly matched teams receive a higher draw probability.
    Large Elo mismatches receive a lower draw probability.
    """

    gap = abs(
        float(adjusted_elo_gap)
    )

    draw_prob = (
        MIN_DRAW_PROB
        + (
            MAX_DRAW_PROB
            - MIN_DRAW_PROB
        )
        * math.exp(
            -gap / DRAW_GAP_SCALE
        )
    )

    return float(
        np.clip(
            draw_prob,
            MIN_DRAW_PROB,
            MAX_DRAW_PROB,
        )
    )


# ============================================================
# 1X2 PROBABILITIES
# ============================================================

def outcome_probabilities(
    home_elo: float,
    away_elo: float,
):
    """
    Convert Elo ratings into HOME / DRAW / AWAY probabilities.

    Home advantage is applied to the matchup strength.

    The Elo win probability is then split around an explicit
    draw probability.
    """

    adjusted_home = (
        home_elo
        + HOME_ADVANTAGE
    )

    adjusted_away = away_elo

    expected_home = elo_expected(
        adjusted_home,
        adjusted_away,
    )

    expected_home = float(
        np.clip(
            expected_home,
            0.01,
            0.99,
        )
    )

    gap = (
        adjusted_home
        - adjusted_away
    )

    draw_prob = draw_probability(
        gap
    )

    remaining = (
        1.0
        - draw_prob
    )

    home_prob = (
        expected_home
        * remaining
    )

    away_prob = (
        (1.0 - expected_home)
        * remaining
    )

    probabilities = np.array(
        [
            home_prob,
            draw_prob,
            away_prob,
        ],
        dtype=float,
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    return (
        float(probabilities[0]),
        float(probabilities[1]),
        float(probabilities[2]),
    )


# ============================================================
# RESULT CLASSIFICATION
# ============================================================

def predicted_result(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
) -> str:
    """
    Select the highest-probability 1X2 outcome.

    DRAW is explicitly allowed.
    """

    probabilities = {
        "HOME": home_prob,
        "DRAW": draw_prob,
        "AWAY": away_prob,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


# ============================================================
# ELO UPDATE
# ============================================================

def update_ratings(
    home_elo: float,
    away_elo: float,
    home_goals: float,
    away_goals: float,
):
    """
    Update ratings after the match.

    IMPORTANT:
    The update happens only after the prediction has been
    generated, preventing future information leakage.
    """

    if home_goals > away_goals:
        actual_home = 1.0

    elif home_goals < away_goals:
        actual_home = 0.0

    else:
        actual_home = 0.5

    expected_home = elo_expected(
        home_elo + HOME_ADVANTAGE,
        away_elo,
    )

    margin = goal_margin_multiplier(
        home_goals - away_goals
    )

    update = (
        K_FACTOR
        * margin
        * (
            actual_home
            - expected_home
        )
    )

    new_home = (
        home_elo
        + update
    )

    new_away = (
        away_elo
        - update
    )

    return (
        new_home,
        new_away,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    input_path = find_input_file()

    print(
        f"Loading strength features from:\n"
        f"{input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    df = prepare_data(
        df
    )

    print(
        f"Matches: {len(df)}"
    )

    print(
        f"Period: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    # --------------------------------------------------------
    # TEAM RATINGS
    # --------------------------------------------------------

    ratings = defaultdict(
        lambda: INITIAL_ELO
    )

    predictions = []

    # --------------------------------------------------------
    # CHRONOLOGICAL PREDICTION
    # --------------------------------------------------------

    for _, row in df.iterrows():

        home = str(
            row["home_team"]
        )

        away = str(
            row["away_team"]
        )

        home_elo = float(
            ratings[home]
        )

        away_elo = float(
            ratings[away]
        )

        # ----------------------------------------------------
        # PRE-MATCH PROBABILITIES
        # ----------------------------------------------------

        (
            home_prob,
            draw_prob,
            away_prob,
        ) = outcome_probabilities(
            home_elo,
            away_elo,
        )

        result = predicted_result(
            home_prob,
            draw_prob,
            away_prob,
        )

        predictions.append(
            {
                "match_id": row["match_id"],
                "date": row["date"],
                "home_team": home,
                "away_team": away,

                "home_elo": home_elo,
                "away_elo": away_elo,

                "elo_difference": (
                    home_elo
                    + HOME_ADVANTAGE
                    - away_elo
                ),

                "home_win_prob": home_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_prob,

                "predicted_result": result,

                "actual_home_goals": row[
                    "home_xg"
                ],
                "actual_away_goals": row[
                    "away_xg"
                ],
            }
        )

        # ----------------------------------------------------
        # UPDATE AFTER PREDICTION
        # ----------------------------------------------------

        # NOTE:
        # epl_strength_features.csv contains xG rather than
        # actual goals, so this version cannot use xG as the
        # Elo update target.
        #
        # Therefore we update using the direction of the
        # observed xG matchup only.
        #
        # This keeps the model chronological but means this
        # Elo model is an xG-strength Elo rather than a
        # conventional goals-based Elo.

        home_xg = float(
            row["home_xg"]
        )

        away_xg = float(
            row["away_xg"]
        )

        new_home, new_away = update_ratings(
            home_elo,
            away_elo,
            home_xg,
            away_xg,
        )

        ratings[home] = new_home
        ratings[away] = new_away

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        predictions
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ELO MODEL")
    print("=" * 70)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Predictions: {len(result)}"
    )

    print()
    print("PREDICTED RESULT DISTRIBUTION")
    print("-" * 70)

    print(
        result[
            "predicted_result"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("SAMPLE PREDICTIONS")
    print("-" * 70)

    display_columns = [
        "date",
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "predicted_result",
    ]

    sample = result[
        display_columns
    ].head(20).copy()

    for column in [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
    ]:

        sample[column] = (
            sample[column]
            * 100
        ).round(1)

    sample["home_elo"] = (
        sample["home_elo"]
        .round(1)
    )

    sample["away_elo"] = (
        sample["away_elo"]
        .round(1)
    )

    print(
        sample.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()