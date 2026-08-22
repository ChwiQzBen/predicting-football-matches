#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

ENSEMBLE_PATH = DATA_DIR / "ensemble_predictions.csv"
SOURCE_PATH = DATA_DIR / "epl_strength_features.csv"

OUTPUT_PATH = DATA_DIR / "ensemble_backtest.csv"


# ============================================================
# CONFIG
# ============================================================

PROB_COLUMNS = [
    "ensemble_home_win_prob",
    "ensemble_draw_prob",
    "ensemble_away_win_prob",
]

RESULT_ORDER = [
    "HOME",
    "DRAW",
    "AWAY",
]


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("Loading ensemble predictions...")

    if not ENSEMBLE_PATH.exists():
        raise FileNotFoundError(
            f"Ensemble predictions not found:\n{ENSEMBLE_PATH}"
        )

    ensemble = pd.read_csv(
        ENSEMBLE_PATH
    )

    print("Loading actual match results...")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Historical match data not found:\n{SOURCE_PATH}"
        )

    source = pd.read_csv(
        SOURCE_PATH
    )

    return ensemble, source


# ============================================================
# NORMALISE
# ============================================================

def prepare_dates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["home_team"] = (
        df["home_team"]
        .astype(str)
        .str.strip()
    )

    df["away_team"] = (
        df["away_team"]
        .astype(str)
        .str.strip()
    )

    return df


def build_match_key(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["date"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        + "|"
        + df["home_team"]
        + "|"
        + df["away_team"]
    )


# ============================================================
# ACTUAL RESULT
# ============================================================

def get_actual_result(
    df: pd.DataFrame,
) -> pd.Series:

    home_goal_columns = [
        "home_goals",
        "home_score",
        "fthg",
        "actual_home_goals",
        "FTHG",
    ]

    away_goal_columns = [
        "away_goals",
        "away_score",
        "ftag",
        "actual_away_goals",
        "FTAG",
    ]

    home_column = next(
        (
            column
            for column in home_goal_columns
            if column in df.columns
        ),
        None,
    )

    away_column = next(
        (
            column
            for column in away_goal_columns
            if column in df.columns
        ),
        None,
    )

    if (
        home_column is not None
        and away_column is not None
    ):

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

    for column in [
        "actual_result",
        "result",
        "outcome",
        "Result",
        "RESULT",
    ]:

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

        return values.map(mapping)

    raise ValueError(
        "No actual result information found in "
        "the historical source dataset."
    )


# ============================================================
# PREPARE BACKTEST DATA
# ============================================================

def prepare_backtest(
    ensemble: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:

    ensemble = prepare_dates(
        ensemble
    )

    source = prepare_dates(
        source
    )

    missing_probabilities = [
        column
        for column in PROB_COLUMNS
        if column not in ensemble.columns
    ]

    if missing_probabilities:

        raise ValueError(
            "Missing ensemble probability columns:\n"
            + "\n".join(
                missing_probabilities
            )
        )

    ensemble["match_key"] = build_match_key(
        ensemble
    )

    source["match_key"] = build_match_key(
        source
    )

    source["actual_result"] = (
        get_actual_result(source)
    )

    actual = source[
        [
            "match_key",
            "actual_result",
        ]
    ].copy()

    actual = actual.drop_duplicates(
        subset=["match_key"],
        keep="last",
    )

    merged = ensemble.merge(
        actual,
        on="match_key",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:

        raise ValueError(
            "No ensemble predictions matched "
            "the historical results."
        )

    merged = merged.dropna(
        subset=[
            "actual_result",
            *PROB_COLUMNS,
        ]
    ).copy()

    if merged.empty:

        raise ValueError(
            "All matched ensemble rows have "
            "missing results or probabilities."
        )

    return merged


# ============================================================
# PROBABILITY PREPARATION
# ============================================================

def get_probabilities(
    df: pd.DataFrame,
) -> np.ndarray:

    probabilities = (
        df[PROB_COLUMNS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float,
            copy=True,
        )
    )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Ensemble probabilities contain "
            "NaN or infinite values."
        )

    if np.nanmean(
        probabilities
    ) > 1.0:

        probabilities = (
            probabilities / 100.0
        )

    probabilities = np.clip(
        probabilities,
        0.0001,
        0.9999,
    )

    row_totals = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_totals <= 0
    ):

        raise ValueError(
            "One or more probability rows "
            "have an invalid total."
        )

    probabilities = (
        probabilities / row_totals
    )

    return probabilities


# ============================================================
# BRIER SCORE
# ============================================================

def calculate_brier_score(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    # probabilities columns are explicitly indexed [HOME, DRAW, AWAY]
    # here (matching PROB_COLUMNS), so this is correct regardless of
    # sklearn's alphabetical-ordering assumption below -- this function
    # never delegates the label<->column correspondence to sklearn.

    actual_home = (
        actual == "HOME"
    ).astype(float)

    actual_draw = (
        actual == "DRAW"
    ).astype(float)

    actual_away = (
        actual == "AWAY"
    ).astype(float)

    squared_error = (
        (
            probabilities[:, 0]
            - actual_home
        ) ** 2
        +
        (
            probabilities[:, 1]
            - actual_draw
        ) ** 2
        +
        (
            probabilities[:, 2]
            - actual_away
        ) ** 2
    )

    return float(
        squared_error.mean()
        / 3.0
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    df: pd.DataFrame,
):

    probabilities = get_probabilities(
        df
    )

    actual = (
        df["actual_result"]
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    # --------------------------------------------------------
    # PREDICTION SOURCE FIX
    #
    # ensemble_predictions.py's add_predictions() can override
    # argmax -- the draw engine may pick DRAW even when it isn't
    # the single largest probability. That decision is saved in
    # the "ensemble_prediction" column of ensemble_predictions.csv.
    #
    # This function used to ignore that column entirely and
    # recompute its own prediction via probabilities.argmax(),
    # which silently threw away every override: a match saved as
    # "DRAW" in ensemble_predictions.csv would get re-decided as
    # HOME or AWAY here, because argmax has no idea an override
    # ever happened. That's why predicted draws could show up in
    # one file and vanish in the other -- two different decision
    # rules were being applied to the same data without anyone
    # choosing that on purpose.
    #
    # Fix: use the saved decision when it's present. Only fall
    # back to argmax if this ensemble_predictions.csv predates the
    # override engine and genuinely doesn't have the column.
    # --------------------------------------------------------

    if "ensemble_prediction" in df.columns:

        predicted = (
            df["ensemble_prediction"]
            .astype(str)
            .str.upper()
            .to_numpy()
        )

        unrecognised = ~np.isin(predicted, RESULT_ORDER)

        if unrecognised.any():
            raise ValueError(
                f"{int(unrecognised.sum())} row(s) in ensemble_prediction "
                f"contain values outside {RESULT_ORDER}: "
                f"{sorted(set(predicted[unrecognised]))}"
            )

    else:

        print(
            "WARNING: ensemble_prediction column not found in "
            "ensemble_predictions.csv -- falling back to argmax(probabilities). "
            "This will NOT reflect any draw override decision. Re-run "
            "ensemble_predictions.py with the current version to get a "
            "file that includes ensemble_prediction."
        )

        predicted = np.array(
            RESULT_ORDER
        )[
            probabilities.argmax(
                axis=1
            )
        ]

    accuracy = accuracy_score(
        actual,
        predicted,
    )

    # --------------------------------------------------------
    # log_loss FIX
    #
    # sklearn's log_loss silently assumes the probability columns
    # are ordered to match the ALPHABETICAL sort of the class
    # labels, regardless of the order given via `labels=`. Passing
    # labels=['HOME','DRAW','AWAY'] does NOT change this -- sklearn
    # still reads column 0 as if it were 'AWAY' (alphabetically
    # first), column 2 as if it were 'HOME'. Our columns are
    # actually built as [HOME, DRAW, AWAY], so HOME and AWAY were
    # being silently swapped in every log_loss call.
    #
    # The real fix: physically reorder the probability columns to
    # match alphabetical order before calling log_loss, and pass
    # labels in that same alphabetical order.
    # --------------------------------------------------------

    alphabetical_order = sorted(RESULT_ORDER)  # ['AWAY', 'DRAW', 'HOME']

    reorder_index = [
        RESULT_ORDER.index(label)
        for label in alphabetical_order
    ]

    probabilities_for_logloss = probabilities[:, reorder_index]

    logloss = log_loss(
        actual,
        probabilities_for_logloss,
        labels=alphabetical_order,
    )

    brier = calculate_brier_score(
        actual,
        probabilities,
    )

    print()
    print("1X2")
    print("-" * 80)

    print(
        f"Accuracy:     {accuracy:.4f}"
    )

    print(
        f"Log Loss:     {logloss:.4f}"
    )

    print(
        f"Brier Score:  {brier:.4f}"
    )

    return (
        probabilities,
        actual,
        predicted,
    )


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

def confidence_analysis(
    probabilities: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
):

    confidence = probabilities.max(
        axis=1
    )

    correct = (
        predicted == actual
    )

    print()
    print(
        "HIGH CONFIDENCE PERFORMANCE"
    )
    print("-" * 80)

    for threshold in [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]:

        mask = (
            confidence >= threshold
        )

        count = int(
            mask.sum()
        )

        if count == 0:

            print(
                f">= {threshold:.0%}: "
                f"0 matches"
            )

            continue

        accuracy = float(
            correct[mask].mean()
        )

        print(
            f">= {threshold:.0%}: "
            f"{count} matches | "
            f"accuracy={accuracy:.4f}"
        )


# ============================================================
# DRAW-SPECIFIC METRICS
# ============================================================

def print_draw_specific_metrics(
    probabilities: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
):

    print()
    print(
        "=" * 80
    )
    print(
        "PER-CLASS METRICS (precision / recall / F1)"
    )
    print(
        "=" * 80
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=RESULT_ORDER,
        zero_division=0,
    )

    print(
        f"{'Class':<8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}"
    )

    for label, p, r, f, s in zip(
        RESULT_ORDER,
        precision,
        recall,
        f1,
        support,
    ):

        print(
            f"{label:<8}{p:>12.4f}{r:>12.4f}{f:>12.4f}{int(s):>12}"
        )

    print()
    print(
        "CONFUSION MATRIX (rows=actual, cols=predicted)"
    )
    print("-" * 80)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=RESULT_ORDER,
    )

    header = " " * 8 + "".join(
        f"{label:>8}" for label in RESULT_ORDER
    )
    print(header)

    for label, row in zip(RESULT_ORDER, matrix):
        row_str = "".join(f"{value:>8}" for value in row)
        print(f"{label:<8}{row_str}")

    # --------------------------------------------------------
    # DRAW as a binary problem: is the draw probability itself
    # well-calibrated, independent of whether DRAW ever wins argmax.
    # --------------------------------------------------------

    draw_index = RESULT_ORDER.index("DRAW")
    draw_prob = probabilities[:, draw_index]
    actual_draw = (actual == "DRAW").astype(int)

    draw_brier = float(
        np.mean((draw_prob - actual_draw) ** 2)
    )

    draw_logloss = log_loss(
        actual_draw,
        np.column_stack([1.0 - draw_prob, draw_prob]),
        labels=[0, 1],
    )

    print()
    print(
        "DRAW PROBABILITY QUALITY (binary: is a draw well-priced, regardless of argmax)"
    )
    print("-" * 80)

    print(
        f"Actual draw rate:      {actual_draw.mean():.4f}"
    )

    print(
        f"Mean predicted draw:   {draw_prob.mean():.4f}"
    )

    print(
        f"Draw Brier:            {draw_brier:.5f}"
    )

    print(
        f"Draw Log Loss:         {draw_logloss:.5f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensemble, source = load_data()

    print(
        f"Ensemble predictions: "
        f"{len(ensemble)}"
    )

    print(
        f"Historical matches:   "
        f"{len(source)}"
    )

    df = prepare_backtest(
        ensemble,
        source,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "CALIBRATED ENSEMBLE BACKTEST"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Evaluation matches: "
        f"{len(df)}"
    )

    print(
        f"Evaluation period: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    (
        probabilities,
        actual,
        predicted,
    ) = print_metrics(df)

    print()
    print(
        "PREDICTED RESULT DISTRIBUTION"
    )

    print("-" * 80)

    print(
        pd.Series(
            predicted,
            name="predicted_result",
        ).value_counts()
        .reindex(
            RESULT_ORDER,
            fill_value=0,
        )
    )

    print()
    print(
        "ACTUAL RESULT DISTRIBUTION"
    )

    print("-" * 80)

    print(
        pd.Series(
            actual,
            name="actual_result",
        ).value_counts()
        .reindex(
            RESULT_ORDER,
            fill_value=0,
        )
    )

    confidence_analysis(
        probabilities,
        actual,
        predicted,
    )

    print_draw_specific_metrics(
        probabilities,
        actual,
        predicted,
    )

    output = df.copy()

    output[
        "predicted_result"
    ] = predicted

    output[
        "actual_result"
    ] = actual

    output[
        "ensemble_home_win_prob"
    ] = probabilities[:, 0]

    output[
        "ensemble_draw_prob"
    ] = probabilities[:, 1]

    output[
        "ensemble_away_win_prob"
    ] = probabilities[:, 2]

    output[
        "ensemble_confidence_probability"
    ] = probabilities.max(
        axis=1
    )

    output[
        "correct"
    ] = (
        output["predicted_result"]
        == output["actual_result"]
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "BACKTEST COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()