#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


SOURCE = Path(
    "data/processed/epl_strength_features.csv"
)


FEATURE_COLUMNS = [
    # --------------------------------------------------
    # HOME ATTACK
    # --------------------------------------------------

    "home_xg_5",
    "home_xg_10",
    "home_xg_diff_5",
    "home_xg_diff_10",
    "home_xg_trend",

    # --------------------------------------------------
    # HOME DEFENCE
    # --------------------------------------------------

    "home_xga_5",
    "home_xga_10",
    "home_xga_trend",

    # --------------------------------------------------
    # HOME VENUE
    # --------------------------------------------------

    "home_venue_xg_5",
    "home_venue_xg_10",
    "home_venue_xga_5",
    "home_venue_xga_10",

    # --------------------------------------------------
    # HOME STRENGTH
    # --------------------------------------------------

    "home_attack_strength_5",
    "home_attack_strength_10",
    "home_defence_strength_5",
    "home_defence_strength_10",

    # --------------------------------------------------
    # HOME FORM / VARIANCE
    # --------------------------------------------------

    "home_xg_std_5",
    "home_xg_std_10",
    "home_finishing_overperformance",
    "home_rest_days",

    # --------------------------------------------------
    # AWAY ATTACK
    # --------------------------------------------------

    "away_xg_5",
    "away_xg_10",
    "away_xg_diff_5",
    "away_xg_diff_10",
    "away_xg_trend",

    # --------------------------------------------------
    # AWAY DEFENCE
    # --------------------------------------------------

    "away_xga_5",
    "away_xga_10",
    "away_xga_trend",

    # --------------------------------------------------
    # AWAY VENUE
    # --------------------------------------------------

    "away_venue_xg_5",
    "away_venue_xg_10",
    "away_venue_xga_5",
    "away_venue_xga_10",

    # --------------------------------------------------
    # AWAY STRENGTH
    # --------------------------------------------------

    "away_attack_strength_5",
    "away_attack_strength_10",
    "away_defence_strength_5",
    "away_defence_strength_10",

    # --------------------------------------------------
    # AWAY FORM / VARIANCE
    # --------------------------------------------------

    "away_xg_std_5",
    "away_xg_std_10",
    "away_finishing_overperformance",
    "away_rest_days",

    # --------------------------------------------------
    # MATCHUP
    # --------------------------------------------------

    "home_matchup_attack",
    "away_matchup_attack",
    "xg_matchup_difference",
    "home_advantage_xg",
]


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "match_id"]
    ).reset_index(drop=True)

    # Require enough historical information.
    df = df[
        (df["home_matches_before"] >= 5)
        & (df["away_matches_before"] >= 5)
    ].copy()

    # Remove rows where model inputs are unavailable.
    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    return df.reset_index(drop=True)


def train_models(
    train: pd.DataFrame,
):

    X = train[FEATURE_COLUMNS]

    y_home = train["home_xg"]
    y_away = train["away_xg"]

    home_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=42,
    )

    away_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=42,
    )

    home_model.fit(
        X,
        y_home,
    )

    away_model.fit(
        X,
        y_away,
    )

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


def baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
):

    """
    Simple baseline.

    Predict the historical mean xG from the training period.
    """

    home_mean = train["home_xg"].mean()
    away_mean = train["away_xg"].mean()

    home_pred = np.full(
        len(test),
        home_mean,
    )

    away_pred = np.full(
        len(test),
        away_mean,
    )

    return home_pred, away_pred


def evaluate_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
):

    home_pred, away_pred = baseline_predictions(
        train,
        test,
    )

    home_mae = mean_absolute_error(
        test["home_xg"],
        home_pred,
    )

    away_mae = mean_absolute_error(
        test["away_xg"],
        away_pred,
    )

    home_rmse = np.sqrt(
        mean_squared_error(
            test["home_xg"],
            home_pred,
        )
    )

    away_rmse = np.sqrt(
        mean_squared_error(
            test["away_xg"],
            away_pred,
        )
    )

    print()
    print("BASELINE")
    print("-" * 50)

    print(
        f"Home xG: "
        f"MAE={home_mae:.4f} "
        f"RMSE={home_rmse:.4f}"
    )

    print(
        f"Away xG: "
        f"MAE={away_mae:.4f} "
        f"RMSE={away_rmse:.4f}"
    )

    return home_pred, away_pred


def main():

    print("Loading strength features...")

    df = pd.read_csv(SOURCE)

    df = prepare_data(df)

    print(
        f"Usable matches: {len(df)}"
    )

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

    # --------------------------------------------------
    # BASELINE
    # --------------------------------------------------

    evaluate_baseline(
        train,
        test,
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print()
    print("TRAINING MATCHUP xG MODELS")
    print("-" * 50)

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

    # --------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------

    test = test.copy()

    test["pred_home_xg"] = home_pred
    test["pred_away_xg"] = away_pred

    output = Path(
        "data/processed/xg_model_predictions.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "pred_home_xg",
        "pred_away_xg",
    ]

    test[columns].to_csv(
        output,
        index=False,
    )

    print()
    print("SAMPLE PREDICTIONS")
    print("-" * 50)

    print(
        test[columns]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        f"Saved predictions: {output}"
    )


if __name__ == "__main__":
    main()
