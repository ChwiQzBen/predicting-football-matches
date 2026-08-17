#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import math

SOURCE = Path(
    "data/processed/calibrated_predictions.csv"
)

OUTPUT = Path(
    "data/processed/final_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# 1X2 confidence thresholds
HIGH_CONFIDENCE = 0.60
MEDIUM_CONFIDENCE = 0.50

# Minimum probability required before recommending a market
MIN_1X2_RECOMMENDATION = 0.50
MIN_GOAL_RECOMMENDATION = 0.55

# Minimum separation between first and second 1X2 outcomes
# used only for confidence classification.
HIGH_MARGIN = 0.15
MEDIUM_MARGIN = 0.08

# Goal markets
GOAL_MARKETS = {
    "OVER 2.5": "cal_over_2_5_prob",
    "BTTS YES": "cal_btts_yes_prob",
}


# ============================================================
# HELPERS
# ============================================================

def safe_probability(value: float) -> float:
    """
    Keep probabilities inside a valid numerical range.
    """

    if pd.isna(value):
        return np.nan

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def fair_odds(
    probability: float,
) -> float:
    """
    Convert a probability into fair decimal odds.

    No bookmaker margin is added.
    """

    probability = safe_probability(
        probability
    )

    if pd.isna(probability) or probability <= 0:
        return np.nan

    return 1.0 / probability


def get_1x2_probabilities(
    row: pd.Series,
) -> dict[str, float]:

    return {
        "HOME": safe_probability(
            row["cal_home_win_prob"]
        ),
        "DRAW": safe_probability(
            row["cal_draw_prob"]
        ),
        "AWAY": safe_probability(
            row["cal_away_win_prob"]
        ),
    }


def get_1x2_prediction(
    row: pd.Series,
) -> tuple[str, float, float]:
    """
    Return:

        prediction
        probability
        margin over second-best outcome
    """

    probabilities = (
        get_1x2_probabilities(row)
    )

    valid = {
        key: value
        for key, value in probabilities.items()
        if pd.notna(value)
    }

    if not valid:
        return "NONE", np.nan, np.nan

    ordered = sorted(
        valid.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    prediction = ordered[0][0]
    probability = ordered[0][1]

    if len(ordered) >= 2:
        margin = (
            ordered[0][1]
            - ordered[1][1]
        )
    else:
        margin = np.nan

    return (
        prediction,
        probability,
        margin,
    )


def classify_confidence(
    probability: float,
    margin: float,
) -> str:
    """
    Confidence reflects both:

    1. Absolute probability.
    2. Separation from the next-best result.
    """

    if pd.isna(probability):
        return "LOW"

    if (
        probability >= HIGH_CONFIDENCE
        and margin >= HIGH_MARGIN
    ):
        return "HIGH"

    if (
        probability >= MEDIUM_CONFIDENCE
        and margin >= MEDIUM_MARGIN
    ):
        return "MEDIUM"

    return "LOW"


def classify_recommendation(
    probability: float,
    margin: float,
) -> str:
    """
    Conservative 1X2 recommendation.

    A result must have both:

    - sufficient probability
    - sufficient separation

    Otherwise PASS.
    """

    if pd.isna(probability):
        return "PASS"

    if (
        probability >= HIGH_CONFIDENCE
        and margin >= HIGH_MARGIN
    ):
        return "STRONG"

    if (
        probability >= MIN_1X2_RECOMMENDATION
        and margin >= MEDIUM_MARGIN
    ):
        return "LEAN"

    return "PASS"


def get_best_goal_market(
    row: pd.Series,
) -> tuple[str, float]:
    """
    Select the highest-probability calibrated goal market.
    """

    markets = {}

    for market, column in GOAL_MARKETS.items():

        if column in row.index:

            value = safe_probability(
                row[column]
            )

            if pd.notna(value):
                markets[market] = value

    if not markets:
        return "NONE", np.nan

    market = max(
        markets,
        key=markets.get,
    )

    return (
        market,
        markets[market],
    )


def classify_goal_recommendation(
    probability: float,
) -> str:
    """
    Conservative goal-market recommendation.
    """

    if pd.isna(probability):
        return "PASS"

    if probability >= 0.65:
        return "STRONG"

    if probability >= MIN_GOAL_RECOMMENDATION:
        return "LEAN"

    return "PASS"


def poisson_score_probability(
    home_xg: float,
    away_xg: float,
    home_goals: int,
    away_goals: int,
) -> float:
    """
    Probability of a specific score using independent
    Poisson goal distributions.

    This is used only to identify the most likely score.
    """

    if (
        pd.isna(home_xg)
        or pd.isna(away_xg)
    ):
        return np.nan

    home_xg = max(
        float(home_xg),
        0.0001,
    )

    away_xg = max(
        float(away_xg),
        0.0001,
    )

    home_probability = (
        np.exp(-home_xg)
        * home_xg ** home_goals
        / math.factorial(home_goals)
    )

    away_probability = (
        np.exp(-away_xg)
        * away_xg ** away_goals
        / math.factorial(away_goals)
    )

    return (
        home_probability
        * away_probability
    )


def get_most_likely_score(
    row: pd.Series,
) -> tuple[str, float]:

    home_xg = row.get(
        "pred_home_xg",
        np.nan,
    )

    away_xg = row.get(
        "pred_away_xg",
        np.nan,
    )

    if (
        pd.isna(home_xg)
        or pd.isna(away_xg)
    ):
        return "N/A", np.nan

    best_score = None
    best_probability = -1.0

    # Evaluate a practical score range.
    for home_goals in range(0, 7):

        for away_goals in range(0, 7):

            probability = (
                poisson_score_probability(
                    home_xg,
                    away_xg,
                    home_goals,
                    away_goals,
                )
            )

            if probability > best_probability:

                best_probability = probability

                best_score = (
                    f"{home_goals}-{away_goals}"
                )

    return (
        best_score,
        best_probability,
    )


# ============================================================
# BUILD FINAL PREDICTIONS
# ============================================================

def build_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # --------------------------------------------------------
    # 1X2
    # --------------------------------------------------------

    predictions = []
    probabilities = []
    margins = []
    confidences = []
    recommendations = []

    for _, row in result.iterrows():

        (
            prediction,
            probability,
            margin,
        ) = get_1x2_prediction(row)

        confidence = (
            classify_confidence(
                probability,
                margin,
            )
        )

        recommendation = (
            classify_recommendation(
                probability,
                margin,
            )
        )

        predictions.append(
            prediction
        )

        probabilities.append(
            probability
        )

        margins.append(
            margin
        )

        confidences.append(
            confidence
        )

        recommendations.append(
            recommendation
        )

    result[
        "predicted_result"
    ] = predictions

    result[
        "prediction_probability"
    ] = probabilities

    result[
        "prediction_margin"
    ] = margins

    result[
        "confidence"
    ] = confidences

    result[
        "recommendation"
    ] = recommendations

    # --------------------------------------------------------
    # INDIVIDUAL 1X2 PROBABILITIES
    # --------------------------------------------------------

    result[
        "fair_home_odds"
    ] = result[
        "cal_home_win_prob"
    ].apply(
        fair_odds
    )

    result[
        "fair_draw_odds"
    ] = result[
        "cal_draw_prob"
    ].apply(
        fair_odds
    )

    result[
        "fair_away_odds"
    ] = result[
        "cal_away_win_prob"
    ].apply(
        fair_odds
    )

    # --------------------------------------------------------
    # PROBABILITY SPREAD
    # --------------------------------------------------------

    result[
        "1x2_probability_spread"
    ] = (
        result[
            [
                "cal_home_win_prob",
                "cal_draw_prob",
                "cal_away_win_prob",
            ]
        ]
        .max(axis=1)
        -
        result[
            [
                "cal_home_win_prob",
                "cal_draw_prob",
                "cal_away_win_prob",
            ]
        ]
        .min(axis=1)
    )

    # --------------------------------------------------------
    # xG
    # --------------------------------------------------------

    result[
        "xg_difference"
    ] = (
        result["pred_home_xg"]
        -
        result["pred_away_xg"]
    )

    result[
        "pred_total_xg"
    ] = (
        result["pred_home_xg"]
        +
        result["pred_away_xg"]
    )

    # --------------------------------------------------------
    # MOST LIKELY SCORE
    # --------------------------------------------------------

    score_values = []
    score_probabilities = []

    for _, row in result.iterrows():

        score, probability = (
            get_most_likely_score(row)
        )

        score_values.append(
            score
        )

        score_probabilities.append(
            probability
        )

    result[
        "most_likely_score"
    ] = score_values

    result[
        "most_likely_score_probability"
    ] = score_probabilities

    # --------------------------------------------------------
    # GOAL MARKET
    # --------------------------------------------------------

    goal_markets = []
    goal_probabilities = []
    goal_recommendations = []

    for _, row in result.iterrows():

        (
            market,
            probability,
        ) = get_best_goal_market(row)

        recommendation = (
            classify_goal_recommendation(
                probability
            )
        )

        goal_markets.append(
            market
        )

        goal_probabilities.append(
            probability
        )

        goal_recommendations.append(
            recommendation
        )

    result[
        "best_goal_market"
    ] = goal_markets

    result[
        "best_goal_market_probability"
    ] = goal_probabilities

    result[
        "goal_market_recommendation"
    ] = goal_recommendations

    # --------------------------------------------------------
    # GOAL MARKET FAIR ODDS
    # --------------------------------------------------------

    result[
        "best_goal_market_fair_odds"
    ] = result[
        "best_goal_market_probability"
    ].apply(
        fair_odds
    )

    # --------------------------------------------------------
    # OVERALL SIGNAL
    #
    # This is deliberately conservative.
    # --------------------------------------------------------

    result[
        "signal_strength"
    ] = np.select(
        [
            (
                (result["recommendation"] == "STRONG")
                &
                (result["goal_market_recommendation"] == "STRONG")
            ),

            (
                (result["recommendation"].isin(
                    ["STRONG", "LEAN"]
                ))
                |
                (result["goal_market_recommendation"].isin(
                    ["STRONG", "LEAN"]
                ))
            ),
        ],
        [
            "STRONG",
            "MODERATE",
        ],
        default="WEAK",
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Loading calibrated predictions..."
    )

    df = pd.read_csv(
        SOURCE
    )

    if df.empty:
        raise ValueError(
            "calibrated_predictions.csv is empty."
        )

    df["date"] = pd.to_datetime(
        df["date"]
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
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "cal_home_win_prob",
        "cal_draw_prob",
        "cal_away_win_prob",
        "cal_over_2_5_prob",
        "cal_btts_yes_prob",
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

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    result = build_predictions(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL PREDICTION SUMMARY")
    print("=" * 70)

    print()
    print("1X2 PREDICTIONS")
    print("-" * 70)

    print(
        result[
            "predicted_result"
        ]
        .value_counts()
        .reindex(
            ["HOME", "DRAW", "AWAY"],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("CONFIDENCE")
    print("-" * 70)

    print(
        result[
            "confidence"
        ]
        .value_counts()
        .reindex(
            ["HIGH", "MEDIUM", "LOW"],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("1X2 RECOMMENDATIONS")
    print("-" * 70)

    print(
        result[
            "recommendation"
        ]
        .value_counts()
        .reindex(
            ["STRONG", "LEAN", "PASS"],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("GOAL MARKET RECOMMENDATIONS")
    print("-" * 70)

    print(
        result[
            "goal_market_recommendation"
        ]
        .value_counts()
        .reindex(
            ["STRONG", "LEAN", "PASS"],
            fill_value=0,
        )
        .to_string()
    )

    print()
    print("SIGNAL STRENGTH")
    print("-" * 70)

    print(
        result[
            "signal_strength"
        ]
        .value_counts()
        .reindex(
            ["STRONG", "MODERATE", "WEAK"],
            fill_value=0,
        )
        .to_string()
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    print()
    print("=" * 120)
    print("FINAL PREDICTIONS")
    print("=" * 120)

    preview_columns = [
        "date",
        "home_team",
        "away_team",
        "pred_home_xg",
        "pred_away_xg",
        "predicted_result",
        "prediction_probability",
        "cal_home_win_prob",
        "cal_draw_prob",
        "cal_away_win_prob",
        "confidence",
        "recommendation",
        "most_likely_score",
        "best_goal_market",
        "best_goal_market_probability",
        "goal_market_recommendation",
        "signal_strength",
    ]

    preview = result[
        preview_columns
    ].head(30).copy()

    # --------------------------------------------------------
    # Display formatting only
    # --------------------------------------------------------

    probability_columns = [
        "prediction_probability",
        "cal_home_win_prob",
        "cal_draw_prob",
        "cal_away_win_prob",
        "best_goal_market_probability",
    ]

    for column in probability_columns:

        preview[column] = (
            preview[column]
            * 100
        ).round(1)

    xg_columns = [
        "pred_home_xg",
        "pred_away_xg",
    ]

    for column in xg_columns:

        preview[column] = (
            preview[column]
            .round(2)
        )

    print(
        preview.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()