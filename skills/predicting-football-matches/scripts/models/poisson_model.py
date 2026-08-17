#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import math
import numpy as np
import pandas as pd


SOURCE = Path(
    "data/processed/xg_model_predictions.csv"
)

OUTPUT = Path(
    "data/processed/poisson_predictions.csv"
)

MAX_GOALS = 10

# Dixon-Coles correlation parameter.
#
# Keep this conservative initially.
# Negative values reduce some low-score combinations
# and positive values increase them.
#
# We will validate this through backtesting rather than
# assuming that a particular value is optimal.
DIXON_COLES_RHO = -0.05


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:

    expected_goals = max(
        float(expected_goals),
        0.001,
    )

    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_xg: float,
    away_xg: float,
    rho: float,
) -> float:

    # Dixon-Coles correction applies only to
    # the four lowest score combinations.

    if home_goals == 0 and away_goals == 0:
        return 1.0 - (
            home_xg
            * away_xg
            * rho
        )

    if home_goals == 0 and away_goals == 1:
        return 1.0 + (
            home_xg * rho
        )

    if home_goals == 1 and away_goals == 0:
        return 1.0 + (
            away_xg * rho
        )

    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho

    return 1.0


def score_matrix(
    home_xg: float,
    away_xg: float,
    rho: float = DIXON_COLES_RHO,
) -> np.ndarray:

    home_xg = max(
        float(home_xg),
        0.001,
    )

    away_xg = max(
        float(away_xg),
        0.001,
    )

    home_probs = np.array(
        [
            poisson_probability(
                i,
                home_xg,
            )
            for i in range(MAX_GOALS + 1)
        ]
    )

    away_probs = np.array(
        [
            poisson_probability(
                i,
                away_xg,
            )
            for i in range(MAX_GOALS + 1)
        ]
    )

    matrix = np.outer(
        home_probs,
        away_probs,
    )

    # Apply Dixon-Coles correction.
    for home_goals in range(
        MAX_GOALS + 1
    ):

        for away_goals in range(
            MAX_GOALS + 1
        ):

            correction = dixon_coles_tau(
                home_goals,
                away_goals,
                home_xg,
                away_xg,
                rho,
            )

            matrix[
                home_goals,
                away_goals,
            ] *= correction

    # Prevent numerical problems.
    matrix = np.clip(
        matrix,
        0.0,
        None,
    )

    # Normalize because:
    # 1. We truncate at MAX_GOALS.
    # 2. Dixon-Coles changes the raw probability mass.
    total = matrix.sum()

    if total <= 0:
        raise ValueError(
            "Score probability matrix has zero probability mass."
        )

    matrix /= total

    return matrix


def calculate_match_probabilities(
    home_xg: float,
    away_xg: float,
) -> dict:

    matrix = score_matrix(
        home_xg,
        away_xg,
    )

    home_win = 0.0
    draw = 0.0
    away_win = 0.0

    over_1_5 = 0.0
    over_2_5 = 0.0
    over_3_5 = 0.0

    btts_yes = 0.0

    for home_goals in range(
        MAX_GOALS + 1
    ):

        for away_goals in range(
            MAX_GOALS + 1
        ):

            probability = matrix[
                home_goals,
                away_goals,
            ]

            if home_goals > away_goals:
                home_win += probability

            elif home_goals == away_goals:
                draw += probability

            else:
                away_win += probability

            total_goals = (
                home_goals
                + away_goals
            )

            if total_goals >= 2:
                over_1_5 += probability

            if total_goals >= 3:
                over_2_5 += probability

            if total_goals >= 4:
                over_3_5 += probability

            if (
                home_goals >= 1
                and away_goals >= 1
            ):
                btts_yes += probability

    # Explicitly normalize 1X2.
    one_x_two_total = (
        home_win
        + draw
        + away_win
    )

    home_win /= one_x_two_total
    draw /= one_x_two_total
    away_win /= one_x_two_total

    # --------------------------------------------------
    # MOST LIKELY SCORE
    # --------------------------------------------------

    best_index = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    most_likely_home = int(
        best_index[0]
    )

    most_likely_away = int(
        best_index[1]
    )

    most_likely_score_probability = float(
        matrix[best_index]
    )

    return {
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,

        "over_1_5_prob": over_1_5,
        "over_2_5_prob": over_2_5,
        "over_3_5_prob": over_3_5,

        "btts_yes_prob": btts_yes,
        "btts_no_prob": 1.0 - btts_yes,

        "expected_total_goals": (
            float(home_xg)
            + float(away_xg)
        ),

        "most_likely_home_goals": (
            most_likely_home
        ),

        "most_likely_away_goals": (
            most_likely_away
        ),

        "most_likely_score_prob": (
            most_likely_score_probability
        ),
    }


def build_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = [
        "pred_home_xg",
        "pred_away_xg",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    results = []

    for _, row in df.iterrows():

        probabilities = (
            calculate_match_probabilities(
                row["pred_home_xg"],
                row["pred_away_xg"],
            )
        )

        results.append(
            probabilities
        )

    probability_df = pd.DataFrame(
        results
    )

    result = pd.concat(
        [
            df.reset_index(drop=True),
            probability_df,
        ],
        axis=1,
    )

    return result


def validate_probabilities(
    df: pd.DataFrame,
) -> None:

    one_x_two = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ]

    totals = one_x_two.sum(
        axis=1
    )

    if not np.allclose(
        totals,
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "1X2 probabilities do not sum to 1."
        )

    probability_columns = [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "over_1_5_prob",
        "over_2_5_prob",
        "over_3_5_prob",
        "btts_yes_prob",
        "btts_no_prob",
    ]

    for column in probability_columns:

        values = df[column].values

        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(
                f"Invalid probability values in {column}."
            )


def main():

    print("Loading xG predictions...")

    df = pd.read_csv(
        SOURCE
    )

    result = build_predictions(
        df
    )

    validate_probabilities(
        result
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"Matches: {len(result)}"
    )

    print(
        f"Saved: {OUTPUT}"
    )

    print()
    print(
        f"Dixon-Coles rho: "
        f"{DIXON_COLES_RHO}"
    )

    print()
    print("SAMPLE PROBABILITIES")
    print("-" * 70)

    preview = [
        "date",
        "home_team",
        "away_team",
        "pred_home_xg",
        "pred_away_xg",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "over_2_5_prob",
        "btts_yes_prob",
        "most_likely_home_goals",
        "most_likely_away_goals",
    ]

    sample = result[
        preview
    ].head(20).copy()

    probability_columns = [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "over_2_5_prob",
        "btts_yes_prob",
    ]

    for column in probability_columns:

        sample[column] = (
            sample[column] * 100
        ).round(1)

    print(
        sample.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()