#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


SOURCE = Path("data/processed/epl_all_features.csv")


FEATURE_COLUMNS = [
    "home_xg_3",
    "home_xg_5",
    "home_xg_10",
    "home_xga_3",
    "home_xga_5",
    "home_xga_10",
    "home_xg_diff_3",
    "home_xg_diff_5",
    "home_xg_diff_10",
    "home_xg_std_5",
    "home_xg_std_10",
    "home_venue_xg_5",
    "home_venue_xg_10",
    "home_venue_xga_5",
    "home_venue_xga_10",
    "home_goal_xg_diff_5",
    "home_goal_xg_diff_10",
    "home_rest_days",

    "away_xg_3",
    "away_xg_5",
    "away_xg_10",
    "away_xga_3",
    "away_xga_5",
    "away_xga_10",
    "away_xg_diff_3",
    "away_xg_diff_5",
    "away_xg_diff_10",
    "away_xg_std_5",
    "away_xg_std_10",
    "away_venue_xg_5",
    "away_venue_xg_10",
    "away_venue_xga_5",
    "away_venue_xga_10",
    "away_goal_xg_diff_5",
    "away_goal_xg_diff_10",
    "away_rest_days",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare leakage-safe training data.

    We require at least five previous matches for both teams.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["home_matches_before"] >= 5)
        & (df["away_matches_before"] >= 5)
    ].copy()

    # Remove rows where required model inputs are unavailable.
    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    return df


def train_models(
    train: pd.DataFrame,
):
    """
    Train separate models for expected home and away xG.

    Targets are Understat's observed match xG, NOT goals.
    """

    X = train[FEATURE_COLUMNS]

    y_home = train["home_xg"]
    y_away = train["away_xg"]

    home_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    away_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    home_model.fit(X, y_home)
    away_model.fit(X, y_away)

    return home_model, away_model


def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
):
    predictions = model.predict(X)

    predictions = np.maximum(
        predictions,
        0.01,
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    print(
        f"{name}: "
        f"MAE={mae:.4f} "
        f"RMSE={rmse:.4f}"
    )

    return predictions


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(SOURCE)

    df = prepare_data(df)

    print(f"Usable matches: {len(df)}")

    # IMPORTANT:
    # Chronological split.
    #
    # Never randomly split football matches.
    #
    # Earlier matches -> training
    # Later matches  -> testing

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    print(
        f"Training matches: {len(train)}"
    )

    print(
        f"Testing matches:  {len(test)}"
    )

    print(
        f"Train period: "
        f"{train['date'].min()} → "
        f"{train['date'].max()}"
    )

    print(
        f"Test period:  "
        f"{test['date'].min()} → "
        f"{test['date'].max()}"
    )

    home_model, away_model = train_models(
        train
    )

    print()
    print("MODEL PERFORMANCE")
    print("-" * 50)

    home_pred = evaluate(
        home_model,
        test[FEATURE_COLUMNS],
        test["home_xg"],
        "Home xG",
    )

    away_pred = evaluate(
        away_model,
        test[FEATURE_COLUMNS],
        test["away_xg"],
        "Away xG",
    )

    test = test.copy()

    test["pred_home_xg"] = home_pred
    test["pred_away_xg"] = away_pred

    print()
    print("SAMPLE PREDICTIONS")
    print("-" * 50)

    columns = [
        "date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "pred_home_xg",
        "pred_away_xg",
    ]

    print(
        test[columns]
        .head(20)
        .to_string(index=False)
    )

    output = Path(
        "data/processed/xg_model_predictions.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test[columns].to_csv(
        output,
        index=False,
    )

    print()
    print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
