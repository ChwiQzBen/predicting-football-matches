#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/processed/poisson_predictions.csv"
)

RESULTS_SOURCE = Path(
    "data/processed/epl_all_matches.csv"
)


def log_loss(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    return float(
        -np.mean(
            outcomes * np.log(probabilities)
            + (1 - outcomes)
            * np.log(1 - probabilities)
        )
    )


def brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:

    return float(
        np.mean(
            (probabilities - outcomes) ** 2
        )
    )


def multiclass_log_loss(
    df: pd.DataFrame,
) -> float:

    probabilities = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].values

    probabilities = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    probabilities /= probabilities.sum(
        axis=1,
        keepdims=True,
    )

    actual = np.where(
        df["home_goals"]
        > df["away_goals"],
        0,
        np.where(
            df["home_goals"]
            == df["away_goals"],
            1,
            2,
        ),
    )

    return float(
        -np.mean(
            np.log(
                probabilities[
                    np.arange(len(df)),
                    actual,
                ]
            )
        )
    )


def multiclass_brier_score(
    df: pd.DataFrame,
) -> float:

    probabilities = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].values

    actual = np.where(
        df["home_goals"]
        > df["away_goals"],
        0,
        np.where(
            df["home_goals"]
            == df["away_goals"],
            1,
            2,
        ),
    )

    actual_one_hot = np.zeros_like(
        probabilities
    )

    actual_one_hot[
        np.arange(len(df)),
        actual,
    ] = 1

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - actual_one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def evaluate_1x2(
    df: pd.DataFrame,
) -> None:

    predicted = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].values

    actual = np.where(
        df["home_goals"]
        > df["away_goals"],
        0,
        np.where(
            df["home_goals"]
            == df["away_goals"],
            1,
            2,
        ),
    )

    predicted_class = np.argmax(
        predicted,
        axis=1,
    )

    accuracy = np.mean(
        predicted_class == actual
    )

    print("1X2")
    print("-" * 50)

    print(
        f"Accuracy:     {accuracy:.4f}"
    )

    print(
        f"Log Loss:     "
        f"{multiclass_log_loss(df):.4f}"
    )

    print(
        f"Brier Score:  "
        f"{multiclass_brier_score(df):.4f}"
    )

    print()


def evaluate_binary(
    df: pd.DataFrame,
    probability_column: str,
    outcome: pd.Series,
    name: str,
) -> None:

    probabilities = (
        df[probability_column]
        .values
    )

    outcomes = (
        outcome
        .astype(int)
        .values
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = np.mean(
        predictions == outcomes
    )

    print(name)
    print("-" * 50)

    print(
        f"Accuracy:     {accuracy:.4f}"
    )

    print(
        f"Log Loss:     "
        f"{log_loss(probabilities, outcomes):.4f}"
    )

    print(
        f"Brier Score:  "
        f"{brier_score(probabilities, outcomes):.4f}"
    )

    print()


def evaluate_xg(
    df: pd.DataFrame,
) -> None:

    home_mae = np.mean(
        np.abs(
            df["pred_home_xg"]
            - df["home_xg"]
        )
    )

    away_mae = np.mean(
        np.abs(
            df["pred_away_xg"]
            - df["away_xg"]
        )
    )

    home_rmse = np.sqrt(
        np.mean(
            (
                df["pred_home_xg"]
                - df["home_xg"]
            ) ** 2
        )
    )

    away_rmse = np.sqrt(
        np.mean(
            (
                df["pred_away_xg"]
                - df["away_xg"]
            ) ** 2
        )
    )

    print("EXPECTED GOALS")
    print("-" * 50)

    print(
        f"Home xG MAE:   {home_mae:.4f}"
    )

    print(
        f"Away xG MAE:   {away_mae:.4f}"
    )

    print(
        f"Home xG RMSE:  {home_rmse:.4f}"
    )

    print(
        f"Away xG RMSE:  {away_rmse:.4f}"
    )

    print()


def calibration_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    probabilities = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ]

    max_probability = probabilities.max(
        axis=1
    )

    predicted_class = probabilities.idxmax(
        axis=1
    )

    actual_class = np.where(
        df["home_goals"]
        > df["away_goals"],
        "home_win_prob",
        np.where(
            df["home_goals"]
            == df["away_goals"],
            "draw_prob",
            "away_win_prob",
        ),
    )

    calibration = pd.DataFrame({
        "confidence": max_probability,
        "correct": (
            predicted_class.values
            == actual_class
        ),
    })

    calibration["bucket"] = pd.cut(
        calibration["confidence"],
        bins=[
            0.0,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
        include_lowest=True,
    )

    result = (
        calibration
        .groupby(
            "bucket",
            observed=True,
        )
        .agg(
            matches=("correct", "size"),
            predicted_probability=(
                "confidence",
                "mean",
            ),
            actual_accuracy=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    return result


def load_data() -> pd.DataFrame:
    """
    Load model predictions and merge them with
    the actual historical match results.
    """

    print("Loading predictions...")

    predictions = pd.read_csv(
        SOURCE
    )

    print("Loading actual results...")

    results = pd.read_csv(
        RESULTS_SOURCE
    )

    # Only bring the actual result fields
    # needed for backtesting.
    results = results[
        [
            "match_id",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    # Prevent accidental duplicate matches.
    if results["match_id"].duplicated().any():

        duplicates = int(
            results["match_id"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Actual results contain "
            f"{duplicates} duplicate match_id values."
        )

    # Merge predictions with actual results.
    df = predictions.merge(
        results,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    # Check that every prediction has an
    # actual result.
    missing_results = df[
        [
            "home_goals",
            "away_goals",
        ]
    ].isna().any(axis=1)

    if missing_results.any():

        missing_count = int(
            missing_results.sum()
        )

        missing_ids = (
            df.loc[
                missing_results,
                "match_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"{missing_count} matches are missing "
            f"actual results. "
            f"Match IDs: {missing_ids[:10]}"
        )

    df["home_goals"] = (
        df["home_goals"]
        .astype(int)
    )

    df["away_goals"] = (
        df["away_goals"]
        .astype(int)
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return df


def main():

    df = load_data()

    print()
    print(
        f"Matches: {len(df)}"
    )

    print()

    # --------------------------------------------------
    # 1X2
    # --------------------------------------------------

    evaluate_1x2(
        df
    )

    # --------------------------------------------------
    # OVER 2.5
    # --------------------------------------------------

    evaluate_binary(
        df,
        "over_2_5_prob",
        (
            df["home_goals"]
            + df["away_goals"]
            > 2.5
        ),
        "OVER 2.5",
    )

    # --------------------------------------------------
    # BTTS
    # --------------------------------------------------

    evaluate_binary(
        df,
        "btts_yes_prob",
        (
            (df["home_goals"] >= 1)
            & (df["away_goals"] >= 1)
        ),
        "BTTS",
    )

    # --------------------------------------------------
    # XG
    # --------------------------------------------------

    evaluate_xg(
        df
    )

    # --------------------------------------------------
    # CONFIDENCE CALIBRATION
    # --------------------------------------------------

    print(
        "CONFIDENCE CALIBRATION"
    )

    print(
        "-" * 50
    )

    calibration = calibration_table(
        df
    )

    print(
        calibration.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()