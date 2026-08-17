#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/processed/calibrated_predictions.csv"
)


def multiclass_metrics(
    df: pd.DataFrame,
    probability_columns: list[str],
) -> dict:

    probabilities = df[
        probability_columns
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
        df["home_goals"] > df["away_goals"],
        0,
        np.where(
            df["home_goals"] == df["away_goals"],
            1,
            2,
        ),
    )

    predicted = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = np.mean(
        predicted == actual
    )

    log_loss = -np.mean(
        np.log(
            probabilities[
                np.arange(len(df)),
                actual,
            ]
        )
    )

    actual_one_hot = np.zeros_like(
        probabilities
    )

    actual_one_hot[
        np.arange(len(df)),
        actual,
    ] = 1

    brier = np.mean(
        np.sum(
            (
                probabilities
                - actual_one_hot
            ) ** 2,
            axis=1,
        )
    )

    return {
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
        "brier": float(brier),
    }


def binary_metrics(
    df: pd.DataFrame,
    probability_column: str,
    outcome: pd.Series,
) -> dict:

    probabilities = np.clip(
        df[probability_column].values,
        1e-15,
        1 - 1e-15,
    )

    outcomes = outcome.astype(int).values

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = np.mean(
        predictions == outcomes
    )

    log_loss = -np.mean(
        outcomes * np.log(probabilities)
        + (1 - outcomes)
        * np.log(1 - probabilities)
    )

    brier = np.mean(
        (probabilities - outcomes) ** 2
    )

    return {
        "accuracy": float(accuracy),
        "log_loss": float(log_loss),
        "brier": float(brier),
    }


def print_comparison(
    name: str,
    raw: dict,
    calibrated: dict,
) -> None:

    print(name)
    print("-" * 70)

    print(
        f"{'Metric':<15}"
        f"{'Raw':>12}"
        f"{'Calibrated':>15}"
        f"{'Change':>15}"
    )

    for metric in [
        "accuracy",
        "log_loss",
        "brier",
    ]:

        raw_value = raw[metric]
        calibrated_value = calibrated[metric]

        change = (
            calibrated_value
            - raw_value
        )

        print(
            f"{metric:<15}"
            f"{raw_value:>12.4f}"
            f"{calibrated_value:>15.4f}"
            f"{change:>15.4f}"
        )

    print()


def main() -> None:

    print("Loading calibrated predictions...")

    df = pd.read_csv(
        SOURCE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Evaluation matches: {len(df)}"
    )

    print(
        f"Evaluation period: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    print()

    # --------------------------------------------------
    # 1X2
    # --------------------------------------------------

    raw_1x2 = multiclass_metrics(
        df,
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ],
    )

    calibrated_1x2 = multiclass_metrics(
        df,
        [
            "cal_home_win_prob",
            "cal_draw_prob",
            "cal_away_win_prob",
        ],
    )

    print_comparison(
        "1X2",
        raw_1x2,
        calibrated_1x2,
    )

    # --------------------------------------------------
    # OVER 2.5
    # --------------------------------------------------

    over_outcome = (
        df["home_goals"]
        + df["away_goals"]
        > 2.5
    )

    raw_over = binary_metrics(
        df,
        "over_2_5_prob",
        over_outcome,
    )

    calibrated_over = binary_metrics(
        df,
        "cal_over_2_5_prob",
        over_outcome,
    )

    print_comparison(
        "OVER 2.5",
        raw_over,
        calibrated_over,
    )

    # --------------------------------------------------
    # BTTS
    # --------------------------------------------------

    btts_outcome = (
        (df["home_goals"] >= 1)
        & (df["away_goals"] >= 1)
    )

    raw_btts = binary_metrics(
        df,
        "btts_yes_prob",
        btts_outcome,
    )

    calibrated_btts = binary_metrics(
        df,
        "cal_btts_yes_prob",
        btts_outcome,
    )

    print_comparison(
        "BTTS",
        raw_btts,
        calibrated_btts,
    )

    # --------------------------------------------------
    # Calibration verdict
    # --------------------------------------------------

    print("=" * 70)
    print("CALIBRATION VERDICT")
    print("=" * 70)

    improvements = 0
    tests = 0

    # Lower is better for log loss and Brier.
    for raw, calibrated in [
        (raw_1x2, calibrated_1x2),
        (raw_over, calibrated_over),
        (raw_btts, calibrated_btts),
    ]:

        tests += 2

        if calibrated["log_loss"] < raw["log_loss"]:
            improvements += 1

        if calibrated["brier"] < raw["brier"]:
            improvements += 1

    print(
        f"Improved scoring metrics: "
        f"{improvements}/{tests}"
    )

    if improvements >= 4:
        print(
            "VERDICT: Calibration is providing "
            "meaningful improvement."
        )

    elif improvements >= 2:
        print(
            "VERDICT: Calibration is mixed. "
            "Keep it under review."
        )

    else:
        print(
            "VERDICT: Calibration is not improving "
            "the model sufficiently."
        )


if __name__ == "__main__":
    main()
