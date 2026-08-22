#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

ELO_PATH = DATA_DIR / "processed" / "elo_predictions.csv"

# Same evaluation period used by the rest of the system.
EVALUATION_START = pd.Timestamp("2026-02-27 20:00:00")


def load_predictions() -> pd.DataFrame:
    if not ELO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find Elo predictions:\n{ELO_PATH}\n\n"
            "Run first:\n"
            "python models/elo_model.py"
        )

    df = pd.read_csv(ELO_PATH)

    required = [
        "date",
        "home_team",
        "away_team",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "actual_home_goals",
        "actual_away_goals",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    numeric_columns = [
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "actual_home_goals",
        "actual_away_goals",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "date",
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
            "actual_home_goals",
            "actual_away_goals",
        ]
    ).copy()

    return df


def add_actual_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["actual_home_goals"] > df["actual_away_goals"],
        df["actual_home_goals"] == df["actual_away_goals"],
        df["actual_home_goals"] < df["actual_away_goals"],
    ]

    df["actual_result"] = np.select(
        conditions,
        ["HOME", "DRAW", "AWAY"],
        default="DRAW",
    )

    return df


def add_predicted_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    probabilities = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].to_numpy()

    labels = np.array(
        [
            "HOME",
            "DRAW",
            "AWAY",
        ]
    )

    df["predicted_result"] = [
        labels[np.argmax(row)]
        for row in probabilities
    ]

    return df


def evaluate_1x2(df: pd.DataFrame):
    y_true = df["actual_result"]

    y_pred = df["predicted_result"]

    probabilities = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].to_numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=["HOME", "DRAW", "AWAY"],
    )

    # Multiclass Brier score.
    actual_matrix = np.zeros_like(
        probabilities
    )

    label_to_index = {
        "HOME": 0,
        "DRAW": 1,
        "AWAY": 2,
    }

    for i, result in enumerate(y_true):
        actual_matrix[
            i,
            label_to_index[result],
        ] = 1.0

    brier = np.mean(
        np.sum(
            (probabilities - actual_matrix) ** 2,
            axis=1,
        )
    )

    return accuracy, logloss, brier


def evaluate_goal_markets(
    df: pd.DataFrame,
):
    df = df.copy()

    actual_total_goals = (
        df["actual_home_goals"]
        + df["actual_away_goals"]
    )

    actual_over = (
        actual_total_goals > 2.5
    ).astype(int)

    predicted_over = (
        df["home_win_prob"] * 0
    )

    # Elo is a 1X2 model.
    #
    # It does NOT generate an independent
    # Over 2.5 or BTTS probability.
    #
    # Therefore we intentionally do not
    # fabricate those probabilities here.

    return actual_over, predicted_over


def confidence_calibration(
    df: pd.DataFrame,
):
    df = df.copy()

    df["max_probability"] = df[
        [
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
        ]
    ].max(axis=1)

    df["correct"] = (
        df["predicted_result"]
        == df["actual_result"]
    ).astype(int)

    bins = [
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    labels = [
        "0.00-0.50",
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    df["confidence_bucket"] = pd.cut(
        df["max_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    calibration = (
        df.groupby(
            "confidence_bucket",
            observed=False,
        )
        .agg(
            matches=("correct", "size"),
            predicted_probability=(
                "max_probability",
                "mean",
            ),
            actual_accuracy=(
                "correct",
                "mean",
            ),
        )
        .reset_index()
    )

    return calibration


def print_result_distribution(
    df: pd.DataFrame,
):
    print()
    print("PREDICTED RESULT DISTRIBUTION")
    print("-" * 70)

    print(
        df["predicted_result"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("ACTUAL RESULT DISTRIBUTION")
    print("-" * 70)

    print(
        df["actual_result"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():
    print("Loading Elo predictions...")

    df = load_predictions()

    print(
        f"Total predictions: {len(df)}"
    )

    print(
        f"Full period: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    # --------------------------------------------------
    # EVALUATION PERIOD
    # --------------------------------------------------

    evaluation = df[
        df["date"] >= EVALUATION_START
    ].copy()

    if evaluation.empty:
        raise ValueError(
            "No matches found in the requested "
            "evaluation period."
        )

    evaluation = add_actual_result(
        evaluation
    )

    evaluation = add_predicted_result(
        evaluation
    )

    print()
    print("=" * 70)
    print("ELO BACKTEST")
    print("=" * 70)

    print(
        f"Evaluation matches: "
        f"{len(evaluation)}"
    )

    print(
        f"Evaluation period: "
        f"{evaluation['date'].min()} → "
        f"{evaluation['date'].max()}"
    )

    # --------------------------------------------------
    # 1X2
    # --------------------------------------------------

    accuracy, logloss, brier = evaluate_1x2(
        evaluation
    )

    print()
    print("1X2")
    print("-" * 50)

    print(
        f"Accuracy:     {accuracy:.4f}"
    )

    print(
        f"Log Loss:     {logloss:.4f}"
    )

    print(
        f"Brier Score:  {brier:.4f}"
    )

    # --------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------

    print_result_distribution(
        evaluation
    )

    # --------------------------------------------------
    # CONFIDENCE CALIBRATION
    # --------------------------------------------------

    print()
    print("CONFIDENCE CALIBRATION")
    print("-" * 50)

    calibration = confidence_calibration(
        evaluation
    )

    calibration["predicted_probability"] = (
        calibration["predicted_probability"]
        .round(4)
    )

    calibration["actual_accuracy"] = (
        calibration["actual_accuracy"]
        .round(4)
    )

    print(
        calibration.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # HIGH CONFIDENCE PERFORMANCE
    # --------------------------------------------------

    evaluation["max_probability"] = (
        evaluation[
            [
                "home_win_prob",
                "draw_prob",
                "away_win_prob",
            ]
        ].max(axis=1)
    )

    print()
    print("HIGH CONFIDENCE PERFORMANCE")
    print("-" * 50)

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:
        subset = evaluation[
            evaluation["max_probability"]
            >= threshold
        ]

        if len(subset) == 0:
            continue

        subset_accuracy = (
            subset["predicted_result"]
            == subset["actual_result"]
        ).mean()

        print(
            f">= {threshold:.0%}: "
            f"{len(subset):3d} matches | "
            f"accuracy={subset_accuracy:.4f}"
        )

    # --------------------------------------------------
    # SAVE BACKTEST DATA
    # --------------------------------------------------

    output = (
        DATA_DIR
        / "processed"
        / "elo_backtest.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
