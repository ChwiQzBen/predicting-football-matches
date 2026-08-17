#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/processed/calibrated_predictions.csv"
)

OUTPUT = Path(
    "data/processed/final_predictions.csv"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_STRONG_PROBABILITY = 0.60
MIN_LEAN_PROBABILITY = 0.50

MIN_STRONG_EDGE = 0.08
MIN_LEAN_EDGE = 0.03


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def classify_confidence(
    probability: float,
) -> str:

    if probability >= MIN_STRONG_PROBABILITY:
        return "HIGH"

    if probability >= MIN_LEAN_PROBABILITY:
        return "MEDIUM"

    return "LOW"


def classify_prediction(
    probability: float,
) -> str:

    if probability >= MIN_STRONG_PROBABILITY:
        return "STRONG"

    if probability >= MIN_LEAN_PROBABILITY:
        return "LEAN"

    return "PASS"


def get_1x2_prediction(
    row: pd.Series,
) -> tuple[str, float]:

    probabilities = {
        "HOME": float(
            row["cal_home_win_prob"]
        ),
        "DRAW": float(
            row["cal_draw_prob"]
        ),
        "AWAY": float(
            row["cal_away_win_prob"]
        ),
    }

    prediction = max(
        probabilities,
        key=probabilities.get,
    )

    probability = probabilities[
        prediction
    ]

    return prediction, probability


def get_best_goal_market(
    row: pd.Series,
) -> tuple[str, float]:

    markets = {
        "OVER 1.5": float(
            row["over_1_5_prob"]
        )
        if "over_1_5_prob" in row.index
        else np.nan,

        "OVER 2.5": float(
            row["cal_over_2_5_prob"]
        ),

        "OVER 3.5": float(
            row["over_3_5_prob"]
        )
        if "over_3_5_prob" in row.index
        else np.nan,

        "BTTS YES": float(
            row["cal_btts_yes_prob"]
        ),
    }

    markets = {
        key: value
        for key, value in markets.items()
        if pd.notna(value)
    }

    if not markets:
        return "NONE", np.nan

    prediction = max(
        markets,
        key=markets.get,
    )

    return (
        prediction,
        markets[prediction],
    )


def build_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    predictions = []
    probabilities = []
    confidences = []
    classifications = []

    goal_markets = []
    goal_market_probs = []

    for _, row in result.iterrows():

        prediction, probability = (
            get_1x2_prediction(row)
        )

        predictions.append(
            prediction
        )

        probabilities.append(
            probability
        )

        confidences.append(
            classify_confidence(
                probability
            )
        )

        classifications.append(
            classify_prediction(
                probability
            )
        )

        market, market_probability = (
            get_best_goal_market(row)
        )

        goal_markets.append(
            market
        )

        goal_market_probs.append(
            market_probability
        )

    result[
        "final_1x2_prediction"
    ] = predictions

    result[
        "final_1x2_probability"
    ] = probabilities

    result[
        "confidence"
    ] = confidences

    result[
        "recommendation"
    ] = classifications

    result[
        "best_goal_market"
    ] = goal_markets

    result[
        "best_goal_market_probability"
    ] = goal_market_probs

    # --------------------------------------------------------
    # Model disagreement
    #
    # Difference between the strongest and weakest 1X2
    # probabilities. This helps identify genuinely decisive
    # matches versus closely balanced matches.
    # --------------------------------------------------------

    probability_columns = [
        "cal_home_win_prob",
        "cal_draw_prob",
        "cal_away_win_prob",
    ]

    probabilities_matrix = result[
        probability_columns
    ].values

    result[
        "1x2_probability_spread"
    ] = (
        probabilities_matrix.max(axis=1)
        - probabilities_matrix.min(axis=1)
    )

    # --------------------------------------------------------
    # xG dominance
    # --------------------------------------------------------

    result[
        "xg_difference"
    ] = (
        result["pred_home_xg"]
        - result["pred_away_xg"]
    )

    result[
        "pred_total_xg"
    ] = (
        result["pred_home_xg"]
        + result["pred_away_xg"]
    )

    # --------------------------------------------------------
    # Most likely score
    #
    # We don't currently carry the full Poisson score matrix
    # into calibrated_predictions.csv, so derive a practical
    # score estimate from the predicted xG.
    #
    # Rounded expected goals are deliberately NOT presented
    # as a probability. They are only a score indication.
    # --------------------------------------------------------

    result[
        "predicted_home_goals"
    ] = (
        result["pred_home_xg"]
        .round()
        .astype(int)
    )

    result[
        "predicted_away_goals"
    ] = (
        result["pred_away_xg"]
        .round()
        .astype(int)
    )

    result[
        "predicted_score"
    ] = (
        result["predicted_home_goals"]
        .astype(str)
        + "-"
        + result["predicted_away_goals"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Overall model signal
    # --------------------------------------------------------

    result[
        "signal_strength"
    ] = np.select(
        [
            (
                (result["final_1x2_probability"]
                 >= MIN_STRONG_PROBABILITY)
                &
                (result["1x2_probability_spread"]
                 >= 0.25)
            ),

            (
                (result["final_1x2_probability"]
                 >= MIN_LEAN_PROBABILITY)
                &
                (result["1x2_probability_spread"]
                 >= 0.15)
            ),
        ],
        [
            "STRONG",
            "MODERATE",
        ],
        default="WEAK",
    )

    return result


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    print("Loading calibrated predictions...")

    df = pd.read_csv(
        SOURCE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Matches: {len(df)}"
    )

    result = build_predictions(
        df
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
        f"Saved: {OUTPUT}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("PREDICTION SUMMARY")
    print("-" * 80)

    print(
        result[
            "final_1x2_prediction"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("CONFIDENCE")
    print("-" * 80)

    print(
        result[
            "confidence"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("RECOMMENDATIONS")
    print("-" * 80)

    print(
        result[
            "recommendation"
        ]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print()
    print("FINAL PREDICTIONS")
    print("-" * 110)

    preview_columns = [
        "date",
        "home_team",
        "away_team",
        "pred_home_xg",
        "pred_away_xg",
        "final_1x2_prediction",
        "final_1x2_probability",
        "confidence",
        "recommendation",
        "predicted_score",
        "best_goal_market",
        "best_goal_market_probability",
    ]

    preview = result[
        preview_columns
    ].head(30).copy()

    probability_columns = [
        "final_1x2_probability",
        "best_goal_market_probability",
    ]

    for column in probability_columns:

        preview[column] = (
            preview[column]
            * 100
        ).round(1)

    preview[
        "pred_home_xg"
    ] = preview[
        "pred_home_xg"
    ].round(2)

    preview[
        "pred_away_xg"
    ] = preview[
        "pred_away_xg"
    ].round(2)

    print(
        preview.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
