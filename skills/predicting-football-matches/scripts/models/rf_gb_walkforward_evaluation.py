#!/usr/bin/env python3
"""
rf_gb_walkforward_evaluation.py

Walk-forward (expanding window) evaluation of RF/GB across the
FULL epl_strength_features.csv history (~1900 matches), not just
the single 1791/109 train/test split used elsewhere.

WHY WALK-FORWARD, NOT "JUST TRAIN ON ALL 1900 AND CHECK ACCURACY":
training on a match and then evaluating the model on that same
match isn't a real test -- the model has already seen the answer.
Walk-forward avoids this: for each chronological slice of matches,
the model is trained ONLY on matches strictly before that slice,
then evaluated on it, then rolled forward. Every match (after an
initial burn-in window) ends up genuinely out-of-sample exactly
once, evaluated by a model that never saw it.

This produces a much larger genuinely-out-of-sample set than the
single 109-match holdout used elsewhere (~1400 matches here), with
enough statistical power to break results down by outcome class,
match closeness, and time -- not just report one overall number.

This script does NOT overwrite model_1x2_random_forest.pkl /
model_1x2_gradient_boosting.pkl. Those stay as whatever
rf_gb_1x2_classifier.py last produced, so ensemble_predictions.py's
backtest against the current 109-match window stays valid and
uncontaminated. See the closing note in this script's output for
what retraining the production models on more data would cost you.

Usage:
    python rf_gb_walkforward_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))

from draw_calibration import get_actual_result  # noqa: E402
from rf_gb_1x2_classifier import (  # noqa: E402
    FEATURE_COLUMNS,
    GB_PARAMS,
    RESULT_ORDER,
    RF_PARAMS,
    build_features,
    calculate_brier_score_3class,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

SOURCE_PATH = DATA_DIR / "epl_strength_features.csv"
OUTPUT_PATH = DATA_DIR / "rf_gb_walkforward_predictions.csv"


# ============================================================
# CONFIG
# ============================================================

# Matches before this point are used only for training, never
# evaluated -- a model trained on 20 matches isn't meaningfully
# tested. 500 is a conservative floor given ~1900 total matches.
INITIAL_TRAIN_SIZE = 500

# Each fold evaluates this many matches, trained on everything
# strictly before them, then rolls forward.
FOLD_SIZE = 150

# Bucket edges for the equilibrium (match-closeness) breakdown.
EQUILIBRIUM_BUCKETS = [0.0, 0.90, 0.97, 0.995, 1.0]
EQUILIBRIUM_BUCKET_LABELS = [
    "lopsided (<0.90)",
    "moderate (0.90-0.97)",
    "close (0.97-0.995)",
    "near-even (0.995+)",
]


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

def run_walk_forward(full: pd.DataFrame) -> pd.DataFrame:

    full = full.sort_values("date").reset_index(drop=True)
    n = len(full)

    if n <= INITIAL_TRAIN_SIZE:
        raise ValueError(
            f"Only {n} matches available, but INITIAL_TRAIN_SIZE is "
            f"{INITIAL_TRAIN_SIZE} -- nothing left to evaluate."
        )

    fold_starts = list(range(INITIAL_TRAIN_SIZE, n, FOLD_SIZE))

    print(f"Total matches: {n}")
    print(f"Initial training burn-in: {INITIAL_TRAIN_SIZE} matches (never evaluated)")
    print(f"Fold size: {FOLD_SIZE} matches")
    print(f"Folds: {len(fold_starts)}")
    print(f"Total out-of-sample matches: {n - INITIAL_TRAIN_SIZE}")
    print()

    records = []

    for fold_number, fold_start in enumerate(fold_starts, start=1):

        fold_end = min(fold_start + FOLD_SIZE, n)

        train_slice = full.iloc[:fold_start]
        eval_slice = full.iloc[fold_start:fold_end]

        X_train = train_slice[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_train = train_slice["actual_result"].to_numpy()

        X_eval = eval_slice[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_eval = eval_slice["actual_result"].to_numpy()

        sample_weight = compute_sample_weight("balanced", y_train)

        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train, sample_weight=sample_weight)

        gb = GradientBoostingClassifier(**GB_PARAMS)
        gb.fit(X_train, y_train, sample_weight=sample_weight)

        rf_probabilities_raw = rf.predict_proba(X_eval)
        gb_probabilities_raw = gb.predict_proba(X_eval)

        rf_class_order = list(rf.classes_)
        gb_class_order = list(gb.classes_)

        blended = {}
        for label in RESULT_ORDER:
            rf_column = rf_probabilities_raw[:, rf_class_order.index(label)]
            gb_column = gb_probabilities_raw[:, gb_class_order.index(label)]
            blended[label] = (rf_column + gb_column) / 2.0

        stacked = np.column_stack([blended[label] for label in RESULT_ORDER])
        predicted = np.array(RESULT_ORDER)[stacked.argmax(axis=1)]

        fold_frame = eval_slice[
            ["date", "home_team", "away_team", "actual_result", "equilibrium"]
        ].copy()

        fold_frame["fold"] = fold_number
        fold_frame["home_prob"] = blended["HOME"]
        fold_frame["draw_prob"] = blended["DRAW"]
        fold_frame["away_prob"] = blended["AWAY"]
        fold_frame["predicted_result"] = predicted
        fold_frame["confidence"] = stacked.max(axis=1)
        fold_frame["correct"] = fold_frame["predicted_result"] == fold_frame["actual_result"]

        records.append(fold_frame)

        print(
            f"Fold {fold_number:>2} | trained on {fold_start:>4} matches | "
            f"evaluated {fold_end - fold_start:>3} matches | "
            f"accuracy {accuracy_score(y_eval, predicted):.4f}"
        )

    return pd.concat(records, ignore_index=True)


# ============================================================
# METRICS
# ============================================================

def print_overall_metrics(results: pd.DataFrame):

    actual = results["actual_result"].to_numpy()
    predicted = results["predicted_result"].to_numpy()

    probabilities = {
        "HOME": results["home_prob"].to_numpy(dtype=float),
        "DRAW": results["draw_prob"].to_numpy(dtype=float),
        "AWAY": results["away_prob"].to_numpy(dtype=float),
    }

    accuracy = accuracy_score(actual, predicted)
    brier = calculate_brier_score_3class(actual, probabilities)

    # Same alphabetical-column-order care as everywhere else in
    # this project: reorder columns to match sorted(RESULT_ORDER)
    # before calling log_loss.
    alphabetical_order = sorted(RESULT_ORDER)
    reordered = np.column_stack([probabilities[label] for label in alphabetical_order])
    logloss = log_loss(actual, reordered, labels=alphabetical_order)

    print()
    print("=" * 80)
    print(f"WALK-FORWARD RESULTS ({len(results)} out-of-sample matches)")
    print("=" * 80)

    print()
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Log Loss:    {logloss:.4f}")
    print(f"Brier:       {brier:.4f}")

    print()
    print("PER-CLASS METRICS")
    print("-" * 80)

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=RESULT_ORDER, zero_division=0
    )

    print(f"{'Class':<8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for label, p, r, f, s in zip(RESULT_ORDER, precision, recall, f1, support):
        print(f"{label:<8}{p:>12.4f}{r:>12.4f}{f:>12.4f}{int(s):>12}")

    print()
    print("CONFUSION MATRIX (rows=actual, cols=predicted)")
    print("-" * 80)

    matrix = confusion_matrix(actual, predicted, labels=RESULT_ORDER)
    header = " " * 8 + "".join(f"{label:>8}" for label in RESULT_ORDER)
    print(header)
    for label, row in zip(RESULT_ORDER, matrix):
        row_str = "".join(f"{value:>8}" for value in row)
        print(f"{label:<8}{row_str}")


def print_equilibrium_breakdown(results: pd.DataFrame):

    print()
    print("=" * 80)
    print("WEAK-AREA BREAKDOWN: BY MATCH CLOSENESS (equilibrium)")
    print("=" * 80)
    print(
        "Lower equilibrium = more lopsided match (big strength gap). "
        "Higher = more evenly matched. If draw recall is concentrated "
        "in one bucket, that's where the model needs the most work."
    )

    results = results.copy()
    results["equilibrium_bucket"] = pd.cut(
        results["equilibrium"],
        bins=EQUILIBRIUM_BUCKETS,
        labels=EQUILIBRIUM_BUCKET_LABELS,
        include_lowest=True,
    )

    print()
    header = f"{'Bucket':<24}{'Matches':>9}{'Accuracy':>10}{'DrawRecall':>12}{'ActualDrawRate':>16}"
    print(header)
    print("-" * len(header))

    for bucket_label in EQUILIBRIUM_BUCKET_LABELS:

        bucket = results[results["equilibrium_bucket"] == bucket_label]

        if bucket.empty:
            continue

        accuracy = (bucket["predicted_result"] == bucket["actual_result"]).mean()

        actual_draws = bucket[bucket["actual_result"] == "DRAW"]
        draw_recall = (
            (actual_draws["predicted_result"] == "DRAW").mean()
            if len(actual_draws) > 0
            else float("nan")
        )

        actual_draw_rate = (bucket["actual_result"] == "DRAW").mean()

        draw_recall_str = f"{draw_recall:.4f}" if not np.isnan(draw_recall) else "n/a"

        print(
            f"{bucket_label:<24}{len(bucket):>9}{accuracy:>10.4f}"
            f"{draw_recall_str:>12}{actual_draw_rate:>16.4f}"
        )


def print_time_trend(results: pd.DataFrame):

    print()
    print("=" * 80)
    print("WEAK-AREA BREAKDOWN: OVER TIME (chronological thirds)")
    print("=" * 80)
    print(
        "If accuracy trends down in more recent thirds, that's a sign of "
        "drift -- the league/teams may be changing in ways the model "
        "hasn't caught up with, worth more than a closeness issue."
    )

    results = results.sort_values("date").reset_index(drop=True)
    n = len(results)
    third = n // 3

    segments = [
        ("First third", results.iloc[:third]),
        ("Middle third", results.iloc[third : 2 * third]),
        ("Last third", results.iloc[2 * third :]),
    ]

    print()
    header = f"{'Segment':<14}{'Matches':>9}{'Accuracy':>10}{'DrawRecall':>12}  {'Period':<26}"
    print(header)
    print("-" * len(header))

    for label, segment in segments:

        if segment.empty:
            continue

        accuracy = (segment["predicted_result"] == segment["actual_result"]).mean()

        actual_draws = segment[segment["actual_result"] == "DRAW"]
        draw_recall = (
            (actual_draws["predicted_result"] == "DRAW").mean()
            if len(actual_draws) > 0
            else float("nan")
        )
        draw_recall_str = f"{draw_recall:.4f}" if not np.isnan(draw_recall) else "n/a"

        period = f"{segment['date'].min().date()} to {segment['date'].max().date()}"

        print(f"{label:<14}{len(segment):>9}{accuracy:>10.4f}{draw_recall_str:>12}  {period:<26}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Historical match data not found:\n{SOURCE_PATH}")

    source = pd.read_csv(SOURCE_PATH)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")

    source["actual_result"] = get_actual_result(source)

    features = build_features(source)

    full = pd.concat(
        [source[["date", "home_team", "away_team", "actual_result"]], features], axis=1
    )
    full = full.dropna(subset=["actual_result", *FEATURE_COLUMNS])

    print(f"Matches after dropping missing values: {len(full)} / {len(source)}")
    print()

    results = run_walk_forward(full)

    results.to_csv(OUTPUT_PATH, index=False)

    print_overall_metrics(results)
    print_equilibrium_breakdown(results)
    print_time_trend(results)

    print()
    print("=" * 80)
    print("NOTE ON PRODUCTION MODELS")
    print("=" * 80)
    print(
        "This script trained many temporary RF/GB models to produce the "
        "walk-forward evaluation above -- it did NOT touch "
        "model_1x2_random_forest.pkl or model_1x2_gradient_boosting.pkl. "
        "Those still reflect whatever rf_gb_1x2_classifier.py last trained "
        "on the 1791-match pre-cutoff population, so ensemble_predictions.py "
        "and ensemble_backtest.py's 109-match evaluation window stays valid."
    )
    print(
        "If you DO want to retrain the production models on more data "
        "later, know the cost: training on matches that are currently your "
        "evaluation set would make future ensemble_backtest.py runs against "
        "that window meaningless (the model would have already seen the "
        "answers). The clean way to grow the training set is to wait for "
        "new matches to accumulate beyond the current evaluation window and "
        "extend the cutoff forward, not to fold the existing evaluation "
        "matches into training."
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
