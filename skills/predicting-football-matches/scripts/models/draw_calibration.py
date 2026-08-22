#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

SOURCE_PATH = DATA_DIR / "epl_strength_features.csv"
CALIBRATED_PATH = DATA_DIR / "calibrated_predictions.csv"

OUTPUT_PATH = DATA_DIR / "draw_calibration.csv"
MODEL_PATH = DATA_DIR / "draw_calibrator.pkl"


# ============================================================
# CONFIG
# ============================================================

# Keep the final evaluation period untouched.
#
# Your current ensemble backtest starts:
#
# 2026-02-27
#
# Therefore calibration is trained only BEFORE this date.
#
CALIBRATION_CUTOFF = pd.Timestamp(
    "2026-02-27 20:00:00"
)


# Raw draw model weights.
#
# These are deliberately not overly aggressive.
#
BASE_DRAW_WEIGHT = 0.35
ELO_DRAW_WEIGHT = 0.20
EQUILIBRIUM_WEIGHT = 0.25
LOW_SCORE_WEIGHT = 0.20


# Probability bounds.
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.60


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for column in candidates:

        if column in df.columns:
            return column

    return None


def find_probability_columns(
    df: pd.DataFrame,
) -> tuple[str, str, str]:

    home = find_column(
        df,
        [
            "home_win_prob",
            "home_probability",
            "home_prob",
            "prob_home",
            "p_home",
        ],
    )

    draw = find_column(
        df,
        [
            "draw_prob",
            "draw_probability",
            "prob_draw",
            "p_draw",
        ],
    )

    away = find_column(
        df,
        [
            "away_win_prob",
            "away_probability",
            "away_prob",
            "prob_away",
            "p_away",
        ],
    )

    if not all(
        [
            home,
            draw,
            away,
        ]
    ):

        raise ValueError(
            "Could not find HOME/DRAW/AWAY "
            "probability columns.\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return home, draw, away


# ============================================================
# RESULT DETECTION
# ============================================================

def get_actual_result(
    df: pd.DataFrame,
) -> pd.Series:

    # --------------------------------------------------------
    # Existing result column
    # --------------------------------------------------------

    result_columns = [
        "actual_result",
        "result",
        "outcome",
        "FTR",
        "ftr",
    ]

    for column in result_columns:

        if column not in df.columns:
            continue

        values = (
            df[column]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        mapping = {
            "H": "HOME",
            "D": "DRAW",
            "A": "AWAY",
            "HOME": "HOME",
            "DRAW": "DRAW",
            "AWAY": "AWAY",
        }

        mapped = values.map(mapping)

        if mapped.notna().sum() > 0:

            return mapped

    # --------------------------------------------------------
    # Goal columns
    # --------------------------------------------------------

    home_column = find_column(
        df,
        [
            "home_goals",
            "home_score",
            "fthg",
            "FTHG",
            "actual_home_goals",
        ],
    )

    away_column = find_column(
        df,
        [
            "away_goals",
            "away_score",
            "ftag",
            "FTAG",
            "actual_away_goals",
        ],
    )

    if (
        home_column is None
        or away_column is None
    ):

        raise ValueError(
            "Could not determine actual match results."
        )

    home_goals = pd.to_numeric(
        df[home_column],
        errors="coerce",
    )

    away_goals = pd.to_numeric(
        df[away_column],
        errors="coerce",
    )

    result = np.where(
        home_goals > away_goals,
        "HOME",
        np.where(
            home_goals < away_goals,
            "AWAY",
            "DRAW",
        ),
    )

    result = pd.Series(
        result,
        index=df.index,
    )

    result[
        home_goals.isna()
        | away_goals.isna()
    ] = np.nan

    return result


# ============================================================
# PROBABILITY NORMALISATION
# ============================================================

def normalise_probability_matrix(
    probabilities: np.ndarray,
) -> np.ndarray:

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).copy()

    if probabilities.ndim != 2:
        raise ValueError(
            "Probability matrix must be 2-dimensional."
        )

    if probabilities.shape[1] != 3:
        raise ValueError(
            "Probability matrix must contain "
            "HOME, DRAW and AWAY."
        )

    finite = np.isfinite(
        probabilities
    )

    if not finite.all():
        raise ValueError(
            "Probability matrix contains "
            "NaN or infinite values."
        )

    # Detect percentages.
    if np.nanmedian(
        probabilities
    ) > 1.0:

        probabilities /= 100.0

    probabilities = np.clip(
        probabilities,
        MIN_PROBABILITY,
        1.0,
    )

    totals = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    probabilities /= totals

    return probabilities


# ============================================================
# POISSON HELPERS
# ============================================================

def poisson_probability(
    goals: int,
    expected_goals: np.ndarray,
) -> np.ndarray:

    expected_goals = np.asarray(
        expected_goals,
        dtype=float,
    )

    expected_goals = np.clip(
        expected_goals,
        0.05,
        6.0,
    )

    return (
        np.exp(-expected_goals)
        * np.power(
            expected_goals,
            goals,
        )
        / math.factorial(goals)
    )


def poisson_draw_probability(
    home_xg: np.ndarray,
    away_xg: np.ndarray,
) -> np.ndarray:

    home_xg = np.asarray(
        home_xg,
        dtype=float,
    )

    away_xg = np.asarray(
        away_xg,
        dtype=float,
    )

    home_xg = np.clip(
        home_xg,
        0.05,
        6.0,
    )

    away_xg = np.clip(
        away_xg,
        0.05,
        6.0,
    )

    draw_probability = np.zeros(
        len(home_xg),
        dtype=float,
    )

    # 0-0 through 7-7.
    for goals in range(8):

        draw_probability += (
            poisson_probability(
                goals,
                home_xg,
            )
            *
            poisson_probability(
                goals,
                away_xg,
            )
        )

    return np.clip(
        draw_probability,
        MIN_PROBABILITY,
        MAX_PROBABILITY,
    )


# ============================================================
# XG DETECTION
# ============================================================

def find_xg_columns(
    df: pd.DataFrame,
) -> tuple[str | None, str | None]:

    home_xg = find_column(
        df,
        [
            "home_xg",
            "xg_home",
            "expected_home_goals",
            "home_expected_goals",
            "predicted_home_goals",
            "home_goal_expectancy",
        ],
    )

    away_xg = find_column(
        df,
        [
            "away_xg",
            "xg_away",
            "expected_away_goals",
            "away_expected_goals",
            "predicted_away_goals",
            "away_goal_expectancy",
        ],
    )

    return home_xg, away_xg


# ============================================================
# STRENGTH DETECTION
# ============================================================

def find_strength_columns(
    df: pd.DataFrame,
) -> tuple[str | None, str | None]:

    home_strength = find_column(
        df,
        [
            "home_strength",
            "home_elo",
            "home_rating",
            "home_power",
            "home_team_strength",
        ],
    )

    away_strength = find_column(
        df,
        [
            "away_strength",
            "away_elo",
            "away_rating",
            "away_power",
            "away_team_strength",
        ],
    )

    return home_strength, away_strength


# ============================================================
# EQUILIBRIUM
# ============================================================

def calculate_equilibrium(
    home_strength: np.ndarray,
    away_strength: np.ndarray,
) -> np.ndarray:

    home_strength = np.asarray(
        home_strength,
        dtype=float,
    )

    away_strength = np.asarray(
        away_strength,
        dtype=float,
    )

    difference = np.abs(
        home_strength
        - away_strength
    )

    # Convert arbitrary rating difference
    # into a 0-1 equilibrium score.
    #
    # 0 difference -> 1.0
    # large difference -> approaches 0.
    #
    # scale is intentionally moderate.
    scale = 300.0

    equilibrium = np.exp(
        -difference / scale
    )

    return np.clip(
        equilibrium,
        0.0,
        1.0,
    )


# ============================================================
# LOW-SCORING SIGNAL
# ============================================================

def calculate_low_score_signal(
    home_xg: np.ndarray,
    away_xg: np.ndarray,
) -> np.ndarray:

    total_xg = (
        np.asarray(
            home_xg,
            dtype=float,
        )
        +
        np.asarray(
            away_xg,
            dtype=float,
        )
    )

    total_xg = np.clip(
        total_xg,
        0.1,
        8.0,
    )

    # Low total expected goals
    # increase draw likelihood.
    #
    # Around 2.0 goals -> moderate signal.
    # Around 1.5 -> stronger.
    # Around 3.5+ -> weak.
    signal = np.exp(
        -0.35
        * np.maximum(
            total_xg - 1.5,
            0.0,
        )
    )

    return np.clip(
        signal,
        0.0,
        1.0,
    )


# ============================================================
# RAW DRAW MODEL
# ============================================================

def calculate_raw_draw_probability(
    base_draw: np.ndarray,
    elo_draw: np.ndarray,
    equilibrium: np.ndarray,
    low_score: np.ndarray,
    poisson_draw: np.ndarray | None = None,
) -> np.ndarray:

    base_draw = np.asarray(
        base_draw,
        dtype=float,
    )

    elo_draw = np.asarray(
        elo_draw,
        dtype=float,
    )

    equilibrium = np.asarray(
        equilibrium,
        dtype=float,
    )

    low_score = np.asarray(
        low_score,
        dtype=float,
    )

    if poisson_draw is None:

        poisson_draw = (
            base_draw
        )

    else:

        poisson_draw = np.asarray(
            poisson_draw,
            dtype=float,
        )

    # --------------------------------------------------------
    # Convert all signals to comparable scales.
    #
    # Base/Elo/Poisson are probabilities.
    # Equilibrium/low-score are structural signals.
    # --------------------------------------------------------

    raw = (
        BASE_DRAW_WEIGHT
        * base_draw
        +
        ELO_DRAW_WEIGHT
        * elo_draw
        +
        EQUILIBRIUM_WEIGHT
        * (
            0.50
            * base_draw
            +
            0.50
            * equilibrium
        )
        +
        LOW_SCORE_WEIGHT
        * (
            0.50
            * poisson_draw
            +
            0.50
            * low_score
            * base_draw
        )
    )

    return np.clip(
        raw,
        MIN_PROBABILITY,
        MAX_PROBABILITY,
    )


# ============================================================
# CALIBRATION DATA
# ============================================================

def build_calibration_dataset(
    source: pd.DataFrame,
) -> pd.DataFrame:

    df = source.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df[
        df["date"]
        < CALIBRATION_CUTOFF
    ].copy()

    print()
    print(
        f"Calibration cutoff: "
        f"{CALIBRATION_CUTOFF}"
    )

    print(
        f"Calibration matches: "
        f"{len(df)}"
    )

    if len(df) < 200:

        raise ValueError(
            "Not enough historical matches "
            "for draw calibration."
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    home_column, draw_column, away_column = (
        find_probability_columns(df)
    )

    probabilities = normalise_probability_matrix(
        df[
            [
                home_column,
                draw_column,
                away_column,
            ]
        ].to_numpy(
            dtype=float
        )
    )

    base_home = probabilities[:, 0]
    base_draw = probabilities[:, 1]
    base_away = probabilities[:, 2]

    # --------------------------------------------------------
    # Elo / strength
    # --------------------------------------------------------

    home_strength_column, away_strength_column = (
        find_strength_columns(df)
    )

    if (
        home_strength_column is not None
        and away_strength_column is not None
    ):

        home_strength = pd.to_numeric(
            df[home_strength_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        away_strength = pd.to_numeric(
            df[away_strength_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid_strength = (
            np.isfinite(home_strength)
            &
            np.isfinite(away_strength)
        )

        if not valid_strength.all():

            home_strength = np.where(
                valid_strength,
                home_strength,
                1500.0,
            )

            away_strength = np.where(
                valid_strength,
                away_strength,
                1500.0,
            )

    else:

        # Fall back to probability equilibrium.
        home_strength = base_home
        away_strength = base_away

    equilibrium = calculate_equilibrium(
        home_strength,
        away_strength,
    )

    # --------------------------------------------------------
    # xG
    # --------------------------------------------------------

    home_xg_column, away_xg_column = (
        find_xg_columns(df)
    )

    if (
        home_xg_column is not None
        and away_xg_column is not None
    ):

        home_xg = pd.to_numeric(
            df[home_xg_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        away_xg = pd.to_numeric(
            df[away_xg_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        valid_xg = (
            np.isfinite(home_xg)
            &
            np.isfinite(away_xg)
        )

        # If some xG values are missing,
        # use a neutral league-level fallback.
        home_xg = np.where(
            valid_xg,
            home_xg,
            1.35,
        )

        away_xg = np.where(
            valid_xg,
            away_xg,
            1.10,
        )

    else:

        # Approximate expected goals from
        # the 1X2 probability structure.
        #
        # This is only a fallback.
        home_xg = (
            0.8
            + 1.8 * base_home
        )

        away_xg = (
            0.7
            + 1.6 * base_away
        )

    poisson_draw = poisson_draw_probability(
        home_xg,
        away_xg,
    )

    low_score = calculate_low_score_signal(
        home_xg,
        away_xg,
    )

    raw_draw = calculate_raw_draw_probability(
        base_draw=base_draw,
        elo_draw=base_draw,
        equilibrium=equilibrium,
        low_score=low_score,
        poisson_draw=poisson_draw,
    )

    # --------------------------------------------------------
    # Actual result
    # --------------------------------------------------------

    actual_result = get_actual_result(
        df
    )

    actual_draw = (
        actual_result
        == "DRAW"
    ).astype(int)

    calibration = pd.DataFrame(
        {
            "date": df["date"].values,
            "home_team": df["home_team"].values,
            "away_team": df["away_team"].values,
            "base_draw_prob": base_draw,
            "equilibrium": equilibrium,
            "low_score_signal": low_score,
            "poisson_draw_prob": poisson_draw,
            "raw_draw_prob": raw_draw,
            "actual_result": actual_result.values,
            "actual_draw": actual_draw.values,
        }
    )

    calibration = calibration.dropna(
        subset=[
            "raw_draw_prob",
            "actual_draw",
        ]
    ).copy()

    return calibration


# ============================================================
# FIT CALIBRATOR
# ============================================================

def fit_calibrator(
    calibration: pd.DataFrame,
):

    x = calibration[
        "raw_draw_prob"
    ].to_numpy(
        dtype=float
    )

    y = calibration[
        "actual_draw"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Isotonic regression
    #
    # y_min/y_max prevent pathological 0/1
    # probabilities.
    # --------------------------------------------------------

    calibrator = IsotonicRegression(
        y_min=MIN_PROBABILITY,
        y_max=MAX_PROBABILITY,
        out_of_bounds="clip",
    )

    calibrator.fit(
        x,
        y,
    )

    calibrated = calibrator.predict(
        x
    )

    calibration[
        "calibrated_draw_prob"
    ] = calibrated

    return calibrator, calibration


# ============================================================
# CALIBRATION METRICS
# ============================================================

def print_calibration_metrics(
    calibration: pd.DataFrame,
):

    actual = calibration[
        "actual_draw"
    ].to_numpy(
        dtype=int
    )

    raw = calibration[
        "raw_draw_prob"
    ].to_numpy(
        dtype=float
    )

    calibrated = calibration[
        "calibrated_draw_prob"
    ].to_numpy(
        dtype=float
    )

    raw_brier = brier_score_loss(
        actual,
        raw,
    )

    calibrated_brier = brier_score_loss(
        actual,
        calibrated,
    )

    # Binary log loss.
    raw_logloss = log_loss(
        actual,
        np.column_stack(
            [
                1.0 - raw,
                raw,
            ]
        ),
        labels=[0, 1],
    )

    calibrated_logloss = log_loss(
        actual,
        np.column_stack(
            [
                1.0 - calibrated,
                calibrated,
            ]
        ),
        labels=[0, 1],
    )

    print()
    print(
        "=" * 80
    )
    print(
        "DRAW CALIBRATION RESULTS"
    )
    print(
        "=" * 80
    )

    print()
    print(
        f"Matches:                 {len(actual)}"
    )

    print(
        f"Actual draw rate:        {actual.mean():.4f}"
    )

    print(
        f"Raw draw mean:           {raw.mean():.4f}"
    )

    print(
        f"Calibrated draw mean:    {calibrated.mean():.4f}"
    )

    print()
    print(
        f"Raw Brier:               {raw_brier:.5f}"
    )

    print(
        f"Calibrated Brier:        {calibrated_brier:.5f}"
    )

    print()
    print(
        f"Raw Log Loss:            {raw_logloss:.5f}"
    )

    print(
        f"Calibrated Log Loss:     {calibrated_logloss:.5f}"
    )

    print()
    print(
        "DRAW CALIBRATION TABLE"
    )

    print("-" * 80)

    calibration["bucket"] = pd.cut(
        calibration[
            "raw_draw_prob"
        ],
        bins=[
            0.00,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
            1.00,
        ],
        include_lowest=True,
    )

    table = (
        calibration
        .groupby(
            "bucket",
            observed=True,
        )
        .agg(
            matches=(
                "actual_draw",
                "size",
            ),
            raw_probability=(
                "raw_draw_prob",
                "mean",
            ),
            calibrated_probability=(
                "calibrated_draw_prob",
                "mean",
            ),
            actual_draw_rate=(
                "actual_draw",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    calibrator,
    calibration: pd.DataFrame,
):

    with open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(
            calibrator,
            file,
        )

    calibration.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved calibration data: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved calibrator: "
        f"{MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading historical match data..."
    )

    source = pd.read_csv(
        SOURCE_PATH
    )

    print(
        f"Historical matches: "
        f"{len(source)}"
    )

    calibration = build_calibration_dataset(
        source
    )

    calibrator, calibration = (
        fit_calibrator(
            calibration
        )
    )

    print_calibration_metrics(
        calibration
    )

    save_outputs(
        calibrator,
        calibration,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "DRAW CALIBRATION COMPLETE"
    )
    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
