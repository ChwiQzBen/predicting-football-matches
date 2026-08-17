#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


PREDICTIONS_SOURCE = Path(
    "data/processed/poisson_predictions.csv"
)

ACTUALS_SOURCE = Path(
    "data/processed/epl_all_matches.csv"
)

OUTPUT = Path(
    "data/processed/calibrated_predictions.csv"
)


PROBABILITY_COLUMNS = [
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
]


def actual_1x2(df: pd.DataFrame) -> np.ndarray:

    return np.where(
        df["home_goals"] > df["away_goals"],
        0,
        np.where(
            df["home_goals"] == df["away_goals"],
            1,
            2,
        ),
    )


def normalize_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:

    probabilities = np.clip(
        probabilities,
        1e-6,
        1.0,
    )

    row_sum = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    return probabilities / row_sum


def calibrate_multiclass(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:

    x_train = train_df[
        PROBABILITY_COLUMNS
    ].values

    x_test = test_df[
        PROBABILITY_COLUMNS
    ].values

    x_train = normalize_probabilities(
        x_train
    )

    x_test = normalize_probabilities(
        x_test
    )

    y_train = actual_1x2(
        train_df
    )

    # --------------------------------------------------
    # Log-ratio probability features
    # --------------------------------------------------
    #
    # These allow the calibration model to learn whether
    # the original Poisson probabilities are systematically
    # too aggressive or too conservative.
    #
    # Away probability is used as the reference class.
    # --------------------------------------------------

    train_reference = np.clip(
        x_train[:, 2:3],
        1e-6,
        1.0,
    )

    test_reference = np.clip(
        x_test[:, 2:3],
        1e-6,
        1.0,
    )

    train_features = np.log(
        np.clip(x_train, 1e-6, 1.0)
        / train_reference
    )

    test_features = np.log(
        np.clip(x_test, 1e-6, 1.0)
        / test_reference
    )

    # --------------------------------------------------
    # Multiclass logistic calibration
    # --------------------------------------------------
    #
    # Do NOT pass multi_class here.
    #
    # Newer scikit-learn versions automatically handle
    # the multiclass problem.
    # --------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
    )

    model.fit(
        train_features,
        y_train,
    )

    calibrated = model.predict_proba(
        test_features
    )

    return normalize_probabilities(
        calibrated
    )


def calibrate_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    probability_column: str,
    outcome_column: str,
) -> np.ndarray:

    x_train = np.clip(
        train_df[
            probability_column
        ].values,
        1e-6,
        1 - 1e-6,
    )

    y_train = train_df[
        outcome_column
    ].astype(int).values

    x_test = np.clip(
        test_df[
            probability_column
        ].values,
        1e-6,
        1 - 1e-6,
    )

    calibrator = IsotonicRegression(
        y_min=0.001,
        y_max=0.999,
        out_of_bounds="clip",
    )

    calibrator.fit(
        x_train,
        y_train,
    )

    return calibrator.predict(
        x_test
    )


def main() -> None:

    print("Loading predictions...")

    predictions = pd.read_csv(
        PREDICTIONS_SOURCE
    )

    print("Loading actual results...")

    actuals = pd.read_csv(
        ACTUALS_SOURCE
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    actuals["date"] = pd.to_datetime(
        actuals["date"]
    )

    # --------------------------------------------------
    # Validate required columns
    # --------------------------------------------------

    required_prediction_columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "over_2_5_prob",
        "btts_yes_prob",
    ]

    missing_prediction_columns = [
        column
        for column in required_prediction_columns
        if column not in predictions.columns
    ]

    if missing_prediction_columns:
        raise ValueError(
            "Predictions are missing required columns: "
            + ", ".join(missing_prediction_columns)
        )

    required_actual_columns = [
        "match_id",
        "home_goals",
        "away_goals",
    ]

    missing_actual_columns = [
        column
        for column in required_actual_columns
        if column not in actuals.columns
    ]

    if missing_actual_columns:
        raise ValueError(
            "Actual results are missing required columns: "
            + ", ".join(missing_actual_columns)
        )

    # --------------------------------------------------
    # Merge predictions with actual results
    # --------------------------------------------------

    df = predictions.merge(
        actuals[required_actual_columns],
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Prediction matches: {len(predictions)}"
    )

    print(
        f"Matched actual results: {len(df)}"
    )

    if len(df) != len(predictions):

        missing = (
            len(predictions)
            - len(df)
        )

        print(
            f"WARNING: {missing} predictions "
            "could not be matched to actual results."
        )

    if len(df) < 100:
        raise ValueError(
            "Too few matched matches for calibration."
        )

    # --------------------------------------------------
    # Temporal calibration split
    # --------------------------------------------------
    #
    # IMPORTANT:
    #
    # Calibration only learns from earlier matches.
    # Evaluation is performed on later matches.
    #
    # This prevents future information from leaking into
    # the calibration model.
    # --------------------------------------------------

    split = int(
        len(df) * 0.70
    )

    calibration_df = df.iloc[
        :split
    ].copy()

    evaluation_df = df.iloc[
        split:
    ].copy()

    print()
    print(
        f"Calibration matches: {len(calibration_df)}"
    )

    print(
        f"Evaluation matches:  {len(evaluation_df)}"
    )

    print(
        f"Calibration period: "
        f"{calibration_df['date'].min()} → "
        f"{calibration_df['date'].max()}"
    )

    print(
        f"Evaluation period:  "
        f"{evaluation_df['date'].min()} → "
        f"{evaluation_df['date'].max()}"
    )

    print()

    # --------------------------------------------------
    # Binary outcomes
    # --------------------------------------------------

    calibration_df["over_2_5"] = (
        calibration_df["home_goals"]
        + calibration_df["away_goals"]
        > 2.5
    ).astype(int)

    evaluation_df["over_2_5"] = (
        evaluation_df["home_goals"]
        + evaluation_df["away_goals"]
        > 2.5
    ).astype(int)

    calibration_df["btts"] = (
        (calibration_df["home_goals"] >= 1)
        & (calibration_df["away_goals"] >= 1)
    ).astype(int)

    evaluation_df["btts"] = (
        (evaluation_df["home_goals"] >= 1)
        & (evaluation_df["away_goals"] >= 1)
    ).astype(int)

    # --------------------------------------------------
    # 1X2 calibration
    # --------------------------------------------------

    calibrated_1x2 = calibrate_multiclass(
        calibration_df,
        evaluation_df,
    )

    evaluation_df[
        "cal_home_win_prob"
    ] = calibrated_1x2[:, 0]

    evaluation_df[
        "cal_draw_prob"
    ] = calibrated_1x2[:, 1]

    evaluation_df[
        "cal_away_win_prob"
    ] = calibrated_1x2[:, 2]

    # --------------------------------------------------
    # OVER 2.5 calibration
    # --------------------------------------------------

    evaluation_df[
        "cal_over_2_5_prob"
    ] = calibrate_binary(
        calibration_df,
        evaluation_df,
        "over_2_5_prob",
        "over_2_5",
    )

    # --------------------------------------------------
    # BTTS calibration
    # --------------------------------------------------

    evaluation_df[
        "cal_btts_yes_prob"
    ] = calibrate_binary(
        calibration_df,
        evaluation_df,
        "btts_yes_prob",
        "btts",
    )

    # --------------------------------------------------
    # Save calibrated predictions
    # --------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"Saved: {OUTPUT}"
    )

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    preview_columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_win_prob",
        "cal_home_win_prob",
        "draw_prob",
        "cal_draw_prob",
        "away_win_prob",
        "cal_away_win_prob",
        "over_2_5_prob",
        "cal_over_2_5_prob",
        "btts_yes_prob",
        "cal_btts_yes_prob",
    ]

    print()
    print(
        evaluation_df[
            preview_columns
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()