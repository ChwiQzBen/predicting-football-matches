#!/usr/bin/env python3

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------
# Make draw_calibration.py importable regardless of the
# working directory this script is launched from.
# --------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from draw_calibration import (  # noqa: E402
    CALIBRATION_CUTOFF,
    MODEL_PATH as DRAW_CALIBRATOR_PATH,
    SOURCE_PATH as STRENGTH_FEATURES_PATH,
    MAX_PROBABILITY as CALIBRATOR_MAX_PROBABILITY,
    MIN_PROBABILITY as CALIBRATOR_MIN_PROBABILITY,
    calculate_equilibrium,
    calculate_low_score_signal,
    calculate_raw_draw_probability,
    find_strength_columns,
    find_xg_columns,
    get_actual_result,
    poisson_draw_probability,
)

from rf_gb_1x2_classifier import (  # noqa: E402
    FEATURE_COLUMNS as TREE_FEATURE_COLUMNS,
    GB_MODEL_PATH,
    RF_MODEL_PATH,
    RESULT_ORDER as TREE_RESULT_ORDER,
    build_features as build_tree_features,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

CALIBRATED_PATH = DATA_DIR / "calibrated_predictions.csv"
ELO_PATH = DATA_DIR / "elo_predictions.csv"

OUTPUT_PATH = DATA_DIR / "ensemble_predictions.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# --------------------------------------------------------
# Main probability blend (HOME / AWAY).
#
# Three legs, not two: base (xG/Poisson) + Elo + a tree ensemble
# (RF+GB averaged, from rf_gb_1x2_classifier.py). The RF/GB
# standalone comparison showed a real, consistent improvement over
# base+Elo alone on accuracy, log loss, Brier, AND non-zero draw
# recall -- so it earns a real vote here, not a token one.
#
# These weights are a judgment call, same as the original 70/30
# split was -- NOT derived from a grid search. Optimizing blend
# weights against the current ~109-match evaluation set would risk
# the exact overfitting-to-a-small-eval-set problem the draw
# override threshold tuning already ran into. Revisit once more
# out-of-sample evaluation data has accumulated.
# --------------------------------------------------------

BASE_WEIGHT = 0.40
ELO_WEIGHT = 0.20
TREE_WEIGHT = 0.40

# --------------------------------------------------------
# Draw blend: the isotonic-calibrated draw probability (extensively
# validated earlier in this project) vs. the tree ensemble's own
# draw probability (which the standalone comparison showed produces
# genuinely non-zero draw recall/precision on its own, something
# the calibrated approach alone never did under argmax). The
# calibrated component keeps the larger share since it went through
# more rigorous validation; the tree component is a meaningful but
# not dominant addition.
# --------------------------------------------------------

DRAW_CALIBRATED_WEIGHT = 0.60
DRAW_TREE_WEIGHT = 0.40

# Probability safety bounds
PROBABILITY_FLOOR = 0.0001

# Recommendation thresholds (HOME/AWAY picks)
STRONG_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.50

STRONG_MARGIN = 0.18
MEDIUM_MARGIN = 0.10

# --------------------------------------------------------
# DRAW SIGNAL WEIGHTS
#
# These are fixed, hand-set starting weights -- NOT tuned from
# data. Only the three DRAW OVERRIDE decision thresholds below
# are data-derived (see tune_draw_override_thresholds). If the
# override behaves badly, these weights are the first thing
# worth revisiting, but that requires judgement, not a grid
# search, since there's no single "correct" weighting without
# a labeled preference for precision vs. recall.
# --------------------------------------------------------

DRAW_SIGNAL_CALIBRATED_WEIGHT = 0.45
DRAW_SIGNAL_BALANCE_WEIGHT = 0.25
DRAW_SIGNAL_LOW_SCORE_WEIGHT = 0.15
DRAW_SIGNAL_CLOSENESS_WEIGHT = 0.15

# --------------------------------------------------------
# DRAW OVERRIDE TUNING
#
# Grid search ranges used to derive DRAW_SIGNAL_THRESHOLD,
# DRAW_PROB_THRESHOLD, and DRAW_MARGIN_THRESHOLD from the
# historical pre-cutoff calibration population. Tuned fresh
# on every run so thresholds never go stale relative to the
# calibrator or historical data.
# --------------------------------------------------------

DRAW_SIGNAL_GRID = np.arange(0.20, 0.501, 0.02)
DRAW_PROB_GRID = np.arange(0.15, 0.401, 0.02)
DRAW_MARGIN_GRID = np.arange(0.02, 0.251, 0.02)

# Minimum number of historical matches (after joining to Elo
# coverage) required before trusting the tuned thresholds at all.
MINIMUM_TUNING_MATCHES = 300

# --------------------------------------------------------
# The tuning objective is F-beta with beta=0.5, not plain F1.
# Plain F1 weights precision and recall equally, which on a
# minority class (draws are ~25-28% of matches) tends to select
# thresholds that call DRAW far too often -- high recall, poor
# precision, and a net drop in overall accuracy. Beta=0.5 weights
# precision twice as heavily as recall, so the search won't accept
# a big accuracy hit just to catch a few more draws.
#
# PRECISION_FLOOR is a second, harder safeguard: candidates below
# this precision are excluded outright regardless of F-beta score.
#
# On real match data, a 35% precision floor was NOT enough --
# it's barely above the "just always guess DRAW" baseline
# (draws are ~28% of matches by base rate), so a threshold combo
# could clear it while still converting most predictions to DRAW.
# Raised to 45%: a DRAW call now has to be meaningfully better
# than guessing before the override is allowed to use it.
#
# MAX_OVERRIDE_RATE_MULTIPLIER is the direct fix for that failure
# mode: even a combo that clears the precision floor is rejected
# if it converts too large a share of ALL matches to DRAW. Capped
# at 1.5x the actual historical draw count -- the override can
# reasonably catch somewhat more draws than the naive rate, but
# not turn DRAW into the majority prediction.
# --------------------------------------------------------

FBETA_BETA = 0.5
PRECISION_FLOOR = 0.45
MAX_OVERRIDE_RATE_MULTIPLIER = 1.5


# ============================================================
# PROBABILITY COLUMNS
# ============================================================

PROB_COLUMNS = [
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
]


# ============================================================
# HELPERS
# ============================================================

def find_probability_columns(df: pd.DataFrame) -> list[str]:

    candidates = [
        ["home_win_prob", "draw_prob", "away_win_prob"],
        ["cal_home_win_prob", "cal_draw_prob", "cal_away_win_prob"],
    ]

    for columns in candidates:
        if all(column in df.columns for column in columns):
            return columns

    raise ValueError(
        "Could not find a complete HOME/DRAW/AWAY probability set."
    )


def validate_probabilities(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:

    df = df.copy()

    values = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    if np.nanmean(values) > 1.0:
        values /= 100.0

    values = np.clip(values, PROBABILITY_FLOOR, 1.0)

    totals = values.sum(axis=1, keepdims=True)
    totals = np.where(totals <= 0, 1.0, totals)

    values = values / totals

    df.loc[:, columns] = values

    return df


def build_match_key(df: pd.DataFrame) -> pd.Series:

    dates = pd.to_datetime(df["date"], errors="coerce")
    home = df["home_team"].astype(str).str.strip()
    away = df["away_team"].astype(str).str.strip()

    return dates.dt.strftime("%Y-%m-%d %H:%M:%S") + "|" + home + "|" + away


def reconcile_probabilities(
    raw_home: np.ndarray,
    raw_away: np.ndarray,
    draw: np.ndarray,
) -> np.ndarray:
    """
    Shared reserve-mass reconciliation: draw gets its own budget,
    HOME/AWAY split whatever remains proportionally to their raw
    (pre-draw) strength. Used identically for the live ensemble
    batch and for reconstructing the historical tuning population,
    so threshold tuning sees exactly the same math the live
    predictions will use.
    """

    non_draw_total = raw_home + raw_away
    non_draw_total = np.maximum(non_draw_total, PROBABILITY_FLOOR)

    available_mass = 1.0 - draw

    home = raw_home / non_draw_total * available_mass
    away = raw_away / non_draw_total * available_mass

    probabilities = np.column_stack([home, draw, away])
    probabilities = np.clip(probabilities, PROBABILITY_FLOOR, 1.0)

    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / totals

    return probabilities


def calculate_closeness_score(
    raw_home: np.ndarray,
    raw_away: np.ndarray,
) -> np.ndarray:
    """
    How evenly matched HOME and AWAY are according to the blended
    (base+Elo) model itself, independent of the strength/xG-based
    `equilibrium` signal. 1.0 = dead even, 0.0 = one side totally
    dominant.
    """

    non_draw_total = np.maximum(raw_home + raw_away, PROBABILITY_FLOOR)

    home_share = raw_home / non_draw_total
    away_share = raw_away / non_draw_total

    return 1.0 - np.abs(home_share - away_share)


def calculate_draw_signal(
    calibrated_draw: np.ndarray,
    equilibrium: np.ndarray,
    low_score: np.ndarray,
    closeness: np.ndarray,
) -> np.ndarray:

    return (
        DRAW_SIGNAL_CALIBRATED_WEIGHT * calibrated_draw
        + DRAW_SIGNAL_BALANCE_WEIGHT * equilibrium
        + DRAW_SIGNAL_LOW_SCORE_WEIGHT * low_score
        + DRAW_SIGNAL_CLOSENESS_WEIGHT * closeness
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading calibrated predictions...")
    calibrated = pd.read_csv(CALIBRATED_PATH)

    print("Loading Elo predictions...")
    elo = pd.read_csv(ELO_PATH)

    calibrated_columns = find_probability_columns(calibrated)
    elo_columns = find_probability_columns(elo)

    print()
    print("Calibrated probability columns:")
    print(" ".join(calibrated_columns))

    print()
    print("Elo probability columns:")
    print(" ".join(elo_columns))

    calibrated = validate_probabilities(calibrated, calibrated_columns)
    elo = validate_probabilities(elo, elo_columns)

    return calibrated, elo, calibrated_columns, elo_columns


def load_draw_calibrator():

    print()
    print("Loading calibrated draw model...")

    if not DRAW_CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"Draw calibrator not found:\n{DRAW_CALIBRATOR_PATH}\n"
            "Run draw_calibration.py first."
        )

    with open(DRAW_CALIBRATOR_PATH, "rb") as file:
        calibrator = pickle.load(file)

    print(f"Loaded: {DRAW_CALIBRATOR_PATH}")

    return calibrator


def load_tree_models():

    print("Loading RF/GB tree ensemble...")

    for path in [RF_MODEL_PATH, GB_MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Tree model not found:\n{path}\n"
                "Run rf_gb_1x2_classifier.py first to train and save both models."
            )

    with open(RF_MODEL_PATH, "rb") as file:
        rf_model = pickle.load(file)

    with open(GB_MODEL_PATH, "rb") as file:
        gb_model = pickle.load(file)

    print(f"Loaded: {RF_MODEL_PATH}")
    print(f"Loaded: {GB_MODEL_PATH}")

    return rf_model, gb_model


def load_strength_features() -> pd.DataFrame:

    if not STRENGTH_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Strength/xG feature source not found:\n{STRENGTH_FEATURES_PATH}\n"
            "This file is required to reconstruct the features the draw "
            "calibrator was trained on."
        )

    features = pd.read_csv(STRENGTH_FEATURES_PATH)

    features["match_key"] = build_match_key(features)

    features = features.drop_duplicates(
        subset=["match_key"],
        keep="last",
    )

    return features


def predict_tree_ensemble(
    joined: pd.DataFrame,
    rf_model,
    gb_model,
) -> dict[str, np.ndarray]:
    """
    Applies both RF and GB to `joined` (a dataframe already carrying
    the epl_strength_features.csv-style columns build_tree_features
    needs) and averages their predict_proba() output into one
    tree-ensemble HOME/DRAW/AWAY probability set.

    Uses the exact same build_tree_features() function the models
    were trained with in rf_gb_1x2_classifier.py -- not a
    reimplementation -- so there's no risk of feature drift between
    training and inference.

    NEVER assumes a fixed column order for predict_proba() output.
    sklearn orders columns by model.classes_ (alphabetical for
    string labels: AWAY, DRAW, HOME), not by TREE_RESULT_ORDER --
    this is the same class of bug already found and fixed once in
    ensemble_backtest.py's log_loss call. Handled here by reading
    model.classes_ directly rather than assuming.
    """

    features = build_tree_features(joined)

    missing_features = features[TREE_FEATURE_COLUMNS].isna().any(axis=1)

    if missing_features.any():
        raise ValueError(
            f"{int(missing_features.sum())} match(es) have missing tree "
            "ensemble features after joining to "
            f"{STRENGTH_FEATURES_PATH.name}. Check that xG/strength/"
            "probability columns are populated for these matches."
        )

    X = features[TREE_FEATURE_COLUMNS].to_numpy(dtype=float)

    rf_probabilities_raw = rf_model.predict_proba(X)
    gb_probabilities_raw = gb_model.predict_proba(X)

    rf_class_order = list(rf_model.classes_)
    gb_class_order = list(gb_model.classes_)

    result = {}

    for label in TREE_RESULT_ORDER:

        rf_column = rf_probabilities_raw[:, rf_class_order.index(label)]
        gb_column = gb_probabilities_raw[:, gb_class_order.index(label)]

        result[label] = (rf_column + gb_column) / 2.0

    return result


# ============================================================
# MATCH ALIGNMENT
# ============================================================

def align_models(calibrated: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:

    calibrated = calibrated.copy()
    elo = elo.copy()

    calibrated["match_key"] = build_match_key(calibrated)
    elo["match_key"] = build_match_key(elo)

    calibrated = calibrated.drop_duplicates(subset=["match_key"], keep="last")
    elo = elo.drop_duplicates(subset=["match_key"], keep="last")

    print()
    print(f"Calibrated matches: {len(calibrated)}")
    print(f"Elo matches:        {len(elo)}")

    merged = calibrated.merge(
        elo,
        on="match_key",
        how="inner",
        suffixes=("", "_elo"),
        validate="one_to_one",
    )

    print(f"Elo matched:        {len(merged)}/{len(calibrated)}")

    if merged.empty:
        raise ValueError("No calibrated predictions matched the Elo predictions.")

    return merged


# ============================================================
# CALIBRATED DRAW COMPONENT
#
# Mirrors draw_calibration.build_calibration_dataset() exactly so
# the calibrator sees the same feature it was trained on:
#   - elo_draw is passed as base_draw (matches training call).
#   - strength/xG come from epl_strength_features.csv, with the
#     same fallbacks used at training time.
#
# Returns equilibrium/low_score/poisson_draw alongside the
# calibrated draw probability, since build_ensemble() needs all
# four for the draw signal and diagnostics -- not just the final
# calibrated number.
# ============================================================

def join_to_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins `df` (must have a match_key column) to
    epl_strength_features.csv, raising a clear error if any match
    can't be matched. Extracted as its own function so the join
    happens exactly once per ensemble run and the result is reused
    for both the draw calibrator and the tree ensemble -- previously
    this join was duplicated inline, which is exactly the kind of
    thing that drifts out of sync between two call sites over time.
    """

    features = load_strength_features()

    joined = df[["match_key"]].merge(
        features,
        on="match_key",
        how="left",
        validate="many_to_one",
    )

    feature_columns = joined.columns.difference(["match_key"])
    unmatched = joined[feature_columns].isna().all(axis=1)

    if unmatched.any():
        raise ValueError(
            f"{int(unmatched.sum())} match(es) in the ensemble could not be "
            f"matched to {STRENGTH_FEATURES_PATH.name}. Check that "
            "date/home_team/away_team align between the two sources."
        )

    return joined


def calculate_calibrated_draw_components(
    joined: pd.DataFrame,
    base_home: np.ndarray,
    base_draw: np.ndarray,
    base_away: np.ndarray,
    calibrator,
) -> dict[str, np.ndarray]:

    home_strength_column, away_strength_column = find_strength_columns(joined)

    if home_strength_column is not None and away_strength_column is not None:

        home_strength = pd.to_numeric(
            joined[home_strength_column], errors="coerce"
        ).to_numpy(dtype=float)

        away_strength = pd.to_numeric(
            joined[away_strength_column], errors="coerce"
        ).to_numpy(dtype=float)

        valid_strength = np.isfinite(home_strength) & np.isfinite(away_strength)

        if not valid_strength.all():
            home_strength = np.where(valid_strength, home_strength, 1500.0)
            away_strength = np.where(valid_strength, away_strength, 1500.0)

    else:

        home_strength = base_home
        away_strength = base_away

    equilibrium = calculate_equilibrium(home_strength, away_strength)

    home_xg_column, away_xg_column = find_xg_columns(joined)

    if home_xg_column is not None and away_xg_column is not None:

        home_xg = pd.to_numeric(
            joined[home_xg_column], errors="coerce"
        ).to_numpy(dtype=float)

        away_xg = pd.to_numeric(
            joined[away_xg_column], errors="coerce"
        ).to_numpy(dtype=float)

        valid_xg = np.isfinite(home_xg) & np.isfinite(away_xg)

        home_xg = np.where(valid_xg, home_xg, 1.35)
        away_xg = np.where(valid_xg, away_xg, 1.10)

    else:

        home_xg = 0.8 + 1.8 * base_home
        away_xg = 0.7 + 1.6 * base_away

    poisson_draw = poisson_draw_probability(home_xg, away_xg)

    low_score = calculate_low_score_signal(home_xg, away_xg)

    raw_draw = calculate_raw_draw_probability(
        base_draw=base_draw,
        elo_draw=base_draw,
        equilibrium=equilibrium,
        low_score=low_score,
        poisson_draw=poisson_draw,
    )

    calibrated_draw = calibrator.predict(raw_draw)

    calibrated_draw = np.clip(
        calibrated_draw,
        CALIBRATOR_MIN_PROBABILITY,
        CALIBRATOR_MAX_PROBABILITY,
    )

    return {
        "calibrated_draw": calibrated_draw,
        "equilibrium": equilibrium,
        "low_score": low_score,
        "poisson_draw": poisson_draw,
    }


# ============================================================
# HISTORICAL DATASET FOR THRESHOLD TUNING
#
# Reconstructs the draw engine's inputs (draw_signal, calibrated
# draw probability, prediction margin) for the same pre-cutoff
# population used to fit draw_calibrator.pkl, joined against
# elo_predictions.csv so the HOME/AWAY blend matches what the live
# ensemble actually does. This is what "learned from the historical
# calibration set" is grounded in -- not guessed.
#
# IMPORTANT CAVEAT: this is in-sample. The same matches were used
# to fit the isotonic calibrator, so tuned thresholds are somewhat
# optimistic. Worth re-validating out-of-sample as more results
# accumulate.
# ============================================================

def build_historical_draw_dataset(calibrator, rf_model, gb_model) -> dict:

    features = pd.read_csv(STRENGTH_FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features[features["date"] < CALIBRATION_CUTOFF].copy()

    n_precutoff = len(features)

    prob_columns = find_probability_columns(features)
    features = validate_probabilities(features, prob_columns)

    features["match_key"] = build_match_key(features)
    features = features.drop_duplicates(subset=["match_key"], keep="last")

    elo = pd.read_csv(ELO_PATH)
    elo_columns = find_probability_columns(elo)
    elo = validate_probabilities(elo, elo_columns)
    elo["match_key"] = build_match_key(elo)
    elo = elo.drop_duplicates(subset=["match_key"], keep="last")

    elo_renamed = elo[["match_key", elo_columns[0], elo_columns[1], elo_columns[2]]].rename(
        columns={
            elo_columns[0]: "hist_elo_home",
            elo_columns[1]: "hist_elo_draw",
            elo_columns[2]: "hist_elo_away",
        }
    )

    joined = features.merge(
        elo_renamed,
        on="match_key",
        how="inner",
        validate="many_to_one",
    )

    n_matched = len(joined)

    if n_matched < MINIMUM_TUNING_MATCHES:
        raise ValueError(
            f"Only {n_matched} of {n_precutoff} pre-cutoff historical matches "
            f"could be matched to {ELO_PATH.name} (need at least "
            f"{MINIMUM_TUNING_MATCHES}). The draw override thresholds cannot "
            "be reliably tuned without broader Elo coverage of the historical "
            "population -- check that elo_predictions.csv actually covers "
            "past matches, not just upcoming/evaluation fixtures."
        )

    base_home = joined[prob_columns[0]].to_numpy(dtype=float)
    base_draw = joined[prob_columns[1]].to_numpy(dtype=float)
    base_away = joined[prob_columns[2]].to_numpy(dtype=float)

    home_strength_column, away_strength_column = find_strength_columns(joined)

    if home_strength_column is not None and away_strength_column is not None:

        home_strength = pd.to_numeric(
            joined[home_strength_column], errors="coerce"
        ).to_numpy(dtype=float)

        away_strength = pd.to_numeric(
            joined[away_strength_column], errors="coerce"
        ).to_numpy(dtype=float)

        valid_strength = np.isfinite(home_strength) & np.isfinite(away_strength)

        home_strength = np.where(valid_strength, home_strength, 1500.0)
        away_strength = np.where(valid_strength, away_strength, 1500.0)

    else:

        home_strength = base_home
        away_strength = base_away

    equilibrium = calculate_equilibrium(home_strength, away_strength)

    home_xg_column, away_xg_column = find_xg_columns(joined)

    if home_xg_column is not None and away_xg_column is not None:

        home_xg = pd.to_numeric(
            joined[home_xg_column], errors="coerce"
        ).to_numpy(dtype=float)

        away_xg = pd.to_numeric(
            joined[away_xg_column], errors="coerce"
        ).to_numpy(dtype=float)

        valid_xg = np.isfinite(home_xg) & np.isfinite(away_xg)

        home_xg = np.where(valid_xg, home_xg, 1.35)
        away_xg = np.where(valid_xg, away_xg, 1.10)

    else:

        home_xg = 0.8 + 1.8 * base_home
        away_xg = 0.7 + 1.6 * base_away

    poisson_draw = poisson_draw_probability(home_xg, away_xg)
    low_score = calculate_low_score_signal(home_xg, away_xg)

    raw_draw = calculate_raw_draw_probability(
        base_draw=base_draw,
        elo_draw=base_draw,
        equilibrium=equilibrium,
        low_score=low_score,
        poisson_draw=poisson_draw,
    )

    calibrated_draw = calibrator.predict(raw_draw)
    calibrated_draw = np.clip(
        calibrated_draw, CALIBRATOR_MIN_PROBABILITY, CALIBRATOR_MAX_PROBABILITY
    )

    tree_probabilities = predict_tree_ensemble(joined, rf_model, gb_model)

    raw_home = (
        BASE_WEIGHT * base_home
        + ELO_WEIGHT * joined["hist_elo_home"].to_numpy(dtype=float)
        + TREE_WEIGHT * tree_probabilities["HOME"]
    )
    raw_away = (
        BASE_WEIGHT * base_away
        + ELO_WEIGHT * joined["hist_elo_away"].to_numpy(dtype=float)
        + TREE_WEIGHT * tree_probabilities["AWAY"]
    )

    draw = DRAW_CALIBRATED_WEIGHT * calibrated_draw + DRAW_TREE_WEIGHT * tree_probabilities["DRAW"]
    draw = np.clip(draw, CALIBRATOR_MIN_PROBABILITY, CALIBRATOR_MAX_PROBABILITY)

    closeness = calculate_closeness_score(raw_home, raw_away)

    draw_signal = calculate_draw_signal(
        calibrated_draw=draw,
        equilibrium=equilibrium,
        low_score=low_score,
        closeness=closeness,
    )

    probabilities = reconcile_probabilities(raw_home, raw_away, draw)
    sorted_probs = np.sort(probabilities, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    natural_prediction = np.array(["HOME", "DRAW", "AWAY"])[
        probabilities.argmax(axis=1)
    ]

    actual_result = get_actual_result(joined)

    valid_mask = (
        actual_result.notna().to_numpy()
        & np.isfinite(draw_signal)
        & np.isfinite(calibrated_draw)
        & np.isfinite(margins)
    )

    return {
        "draw_signal": draw_signal[valid_mask],
        "calibrated_draw": calibrated_draw[valid_mask],
        "margin": margins[valid_mask],
        "natural_prediction": natural_prediction[valid_mask],
        "actual_result": actual_result.to_numpy()[valid_mask],
        "n_precutoff": n_precutoff,
        "n_matched_to_elo": n_matched,
        "n_valid": int(valid_mask.sum()),
    }


def tune_draw_override_thresholds(historical: dict) -> dict:

    draw_signal = historical["draw_signal"]
    calibrated_draw = historical["calibrated_draw"]
    margin = historical["margin"]
    natural = historical["natural_prediction"]
    actual = historical["actual_result"]

    n_actual_draws = int(np.sum(actual == "DRAW"))

    max_draw_predictions = int(np.ceil(MAX_OVERRIDE_RATE_MULTIPLIER * n_actual_draws))

    best = None

    for signal_threshold in DRAW_SIGNAL_GRID:

        signal_pass = draw_signal >= signal_threshold

        if not signal_pass.any():
            continue

        for prob_threshold in DRAW_PROB_GRID:

            prob_pass = signal_pass & (calibrated_draw >= prob_threshold)

            if not prob_pass.any():
                continue

            for margin_threshold in DRAW_MARGIN_GRID:

                override_mask = prob_pass & (margin <= margin_threshold)

                predicted = np.where(override_mask, "DRAW", natural)

                n_draw_predictions = int(np.sum(predicted == "DRAW"))

                if n_draw_predictions > max_draw_predictions:
                    # Hard cap: reject any combo that converts too large a
                    # share of ALL matches to DRAW, even if its precision
                    # technically clears the floor. This is what actually
                    # stops the override from becoming the majority
                    # prediction -- precision alone did not catch this on
                    # real match data.
                    continue

                tp = int(np.sum((predicted == "DRAW") & (actual == "DRAW")))
                fp = int(np.sum((predicted == "DRAW") & (actual != "DRAW")))
                fn = int(np.sum((predicted != "DRAW") & (actual == "DRAW")))

                if tp + fp == 0 or tp + fn == 0:
                    continue

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

                if precision < PRECISION_FLOOR:
                    # Hard safeguard: never accept a threshold combo that
                    # calls DRAW more often wrong than right by this much,
                    # no matter how good recall looks.
                    continue

                if precision + recall == 0:
                    continue

                beta_sq = FBETA_BETA ** 2
                denominator = (beta_sq * precision) + recall

                if denominator == 0:
                    continue

                fbeta = (1 + beta_sq) * precision * recall / denominator
                f1 = 2 * precision * recall / (precision + recall)
                accuracy = float(np.mean(predicted == actual))

                candidate = {
                    "signal_threshold": float(signal_threshold),
                    "prob_threshold": float(prob_threshold),
                    "margin_threshold": float(margin_threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "fbeta": fbeta,
                    "accuracy": accuracy,
                    "n_draw_predictions": n_draw_predictions,
                    "n_actual_draws": n_actual_draws,
                    "max_draw_predictions": max_draw_predictions,
                }

                if best is None or (candidate["fbeta"], candidate["accuracy"]) > (
                    best["fbeta"],
                    best["accuracy"],
                ):
                    best = candidate

    if best is None:
        raise ValueError(
            f"No combination of draw override thresholds cleared both the "
            f"{PRECISION_FLOOR:.0%} precision floor and the "
            f"{MAX_OVERRIDE_RATE_MULTIPLIER}x actual-draw-rate cap "
            f"({max_draw_predictions} predictions max, vs {n_actual_draws} "
            "actual draws) on the historical set. This likely means the "
            "draw signal isn't separating draws from non-draws well enough "
            "for a reliable override on this data -- worth checking the "
            "calibrator and feature inputs before loosening either "
            "constraint."
        )

    return best


# ============================================================
# ENSEMBLE
# ============================================================

def build_ensemble(
    merged: pd.DataFrame,
    calibrated_columns: list[str],
    elo_columns: list[str],
    calibrator,
    rf_model,
    gb_model,
) -> pd.DataFrame:

    df = merged.copy()

    base_home = df[calibrated_columns[0]].to_numpy(dtype=float)
    base_draw = df[calibrated_columns[1]].to_numpy(dtype=float)
    base_away = df[calibrated_columns[2]].to_numpy(dtype=float)

    elo_home = df[f"{elo_columns[0]}_elo"].to_numpy(dtype=float)
    elo_away = df[f"{elo_columns[2]}_elo"].to_numpy(dtype=float)

    joined = join_to_strength_features(df)

    draw_components = calculate_calibrated_draw_components(
        joined=joined,
        base_home=base_home,
        base_draw=base_draw,
        base_away=base_away,
        calibrator=calibrator,
    )

    tree_probabilities = predict_tree_ensemble(joined, rf_model, gb_model)

    # --------------------------------------------------------
    # HOME / AWAY: three-way blend (base + Elo + tree ensemble).
    # --------------------------------------------------------

    raw_home = (
        BASE_WEIGHT * base_home
        + ELO_WEIGHT * elo_home
        + TREE_WEIGHT * tree_probabilities["HOME"]
    )
    raw_away = (
        BASE_WEIGHT * base_away
        + ELO_WEIGHT * elo_away
        + TREE_WEIGHT * tree_probabilities["AWAY"]
    )

    # --------------------------------------------------------
    # DRAW: two-way blend (isotonic-calibrated + tree ensemble).
    # This is the piece that actually captures why the tree
    # comparison mattered -- RF/GB showed real non-zero draw
    # recall on their own, something the calibrated-only approach
    # never produced under plain argmax.
    # --------------------------------------------------------

    calibrated_draw = draw_components["calibrated_draw"]
    tree_draw = tree_probabilities["DRAW"]

    draw = DRAW_CALIBRATED_WEIGHT * calibrated_draw + DRAW_TREE_WEIGHT * tree_draw

    draw = np.clip(draw, CALIBRATOR_MIN_PROBABILITY, CALIBRATOR_MAX_PROBABILITY)

    closeness = calculate_closeness_score(raw_home, raw_away)

    draw_signal = calculate_draw_signal(
        calibrated_draw=draw,
        equilibrium=draw_components["equilibrium"],
        low_score=draw_components["low_score"],
        closeness=closeness,
    )

    probabilities = reconcile_probabilities(raw_home, raw_away, draw)

    df["ensemble_home_win_prob"] = probabilities[:, 0] * 100.0
    df["ensemble_draw_prob"] = probabilities[:, 1] * 100.0
    df["ensemble_away_win_prob"] = probabilities[:, 2] * 100.0

    df["draw_signal"] = draw_signal
    df["draw_equilibrium"] = draw_components["equilibrium"]
    df["draw_low_score_signal"] = draw_components["low_score"]
    df["draw_poisson_probability"] = draw_components["poisson_draw"] * 100.0
    df["draw_closeness_score"] = closeness
    df["tree_home_win_prob"] = tree_probabilities["HOME"] * 100.0
    df["tree_draw_prob"] = tree_probabilities["DRAW"] * 100.0
    df["tree_away_win_prob"] = tree_probabilities["AWAY"] * 100.0

    return df


# ============================================================
# PREDICTION / CONFIDENCE
#
# DRAW no longer has to win argmax. The override test fires when
# draw_signal, the calibrated draw probability, AND a tight
# HOME/AWAY margin all agree -- three independent conditions,
# not one. When it fires, "confidence" and "margin" are redefined
# relative to the PREDICTED class rather than always the max
# probability, since those two can now diverge: an override pick
# can legitimately have a negative margin (it wasn't the biggest
# probability, threshold logic put it there anyway). That's
# intentional -- it's how an overridden pick is distinguished from
# a probability-dominant one downstream.
# ============================================================

def add_predictions(
    df: pd.DataFrame,
    override_thresholds: dict,
) -> pd.DataFrame:

    df = df.copy()

    probabilities = (
        df[["ensemble_home_win_prob", "ensemble_draw_prob", "ensemble_away_win_prob"]]
        .to_numpy(dtype=float)
        / 100.0
    )

    result_names = np.array(["HOME", "DRAW", "AWAY"])

    natural_winner_index = probabilities.argmax(axis=1)
    natural_predictions = result_names[natural_winner_index]

    sorted_probabilities = np.sort(probabilities, axis=1)
    natural_margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    draw_signal = df["draw_signal"].to_numpy(dtype=float)
    calibrated_draw = probabilities[:, 1]

    override_mask = (
        (draw_signal >= override_thresholds["signal"])
        & (calibrated_draw >= override_thresholds["prob"])
        & (natural_margins <= override_thresholds["margin"])
    )

    predictions = np.where(override_mask, "DRAW", natural_predictions)

    draw_override_applied = override_mask & (natural_predictions != "DRAW")

    predicted_index = np.array(
        [{"HOME": 0, "DRAW": 1, "AWAY": 2}[label] for label in predictions]
    )

    row_index = np.arange(len(predictions))

    predicted_confidence = probabilities[row_index, predicted_index]

    other_mask = np.ones_like(probabilities, dtype=bool)
    other_mask[row_index, predicted_index] = False

    best_other = np.where(other_mask, probabilities, -np.inf).max(axis=1)

    predicted_margin = predicted_confidence - best_other

    confidence_labels = np.where(
        (predicted_confidence >= STRONG_THRESHOLD) & (predicted_margin >= STRONG_MARGIN),
        "HIGH",
        np.where(
            (predicted_confidence >= MEDIUM_THRESHOLD) & (predicted_margin >= MEDIUM_MARGIN),
            "MEDIUM",
            "LOW",
        ),
    )

    draw_prob_fraction = probabilities[:, 1]

    recommendation = np.where(
        (predictions == "DRAW") & (draw_prob_fraction >= 0.30),
        "STRONG",
        np.where(
            (predicted_confidence >= STRONG_THRESHOLD) & (predicted_margin >= STRONG_MARGIN),
            "STRONG",
            np.where(
                (predicted_confidence >= MEDIUM_THRESHOLD) & (predicted_margin >= MEDIUM_MARGIN),
                "LEAN",
                "PASS",
            ),
        ),
    )

    df["ensemble_prediction"] = predictions
    df["ensemble_probability"] = predicted_confidence * 100.0
    df["prediction_margin"] = predicted_margin * 100.0
    df["ensemble_confidence"] = confidence_labels
    df["ensemble_recommendation"] = recommendation
    df["draw_override_applied"] = draw_override_applied

    return df


# ============================================================
# OUTPUT
# ============================================================

def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:

    preferred = [
        "match_id",
        "date",
        "home_team",
        "away_team",
        "ensemble_home_win_prob",
        "ensemble_draw_prob",
        "ensemble_away_win_prob",
        "ensemble_prediction",
        "ensemble_probability",
        "prediction_margin",
        "ensemble_confidence",
        "ensemble_recommendation",
        "draw_signal",
        "draw_equilibrium",
        "draw_low_score_signal",
        "draw_poisson_probability",
        "draw_closeness_score",
        "draw_override_applied",
        "tree_home_win_prob",
        "tree_draw_prob",
        "tree_away_win_prob",
    ]

    available = [column for column in preferred if column in df.columns]

    return df[available].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    calibrated, elo, calibrated_columns, elo_columns = load_data()

    calibrator = load_draw_calibrator()

    rf_model, gb_model = load_tree_models()

    print()
    print("Tuning draw override thresholds against historical calibration set...")

    historical = build_historical_draw_dataset(calibrator, rf_model, gb_model)

    print(f"Pre-cutoff historical matches: {historical['n_precutoff']}")
    print(f"Matched to Elo predictions:    {historical['n_matched_to_elo']}")
    print(f"Valid for tuning:              {historical['n_valid']}")

    best_thresholds = tune_draw_override_thresholds(historical)

    override_thresholds = {
        "signal": best_thresholds["signal_threshold"],
        "prob": best_thresholds["prob_threshold"],
        "margin": best_thresholds["margin_threshold"],
    }

    print()
    print("DRAW OVERRIDE THRESHOLDS (tuned on historical data)")
    print("-" * 80)
    print(f"Signal threshold:           {override_thresholds['signal']:.2f}")
    print(f"Draw probability threshold: {override_thresholds['prob']:.2f}")
    print(f"Margin threshold:           {override_thresholds['margin']:.2f}")
    print()
    print(f"In-sample DRAW precision:   {best_thresholds['precision']:.4f}")
    print(f"In-sample DRAW recall:      {best_thresholds['recall']:.4f}")
    print(f"In-sample DRAW F1:          {best_thresholds['f1']:.4f}")
    print(f"In-sample DRAW F-beta(0.5): {best_thresholds['fbeta']:.4f}  (tuning objective)")
    print(f"In-sample overall accuracy: {best_thresholds['accuracy']:.4f}")
    print(
        f"In-sample draw predictions: {best_thresholds['n_draw_predictions']} "
        f"(actual draws: {best_thresholds['n_actual_draws']}, "
        f"cap: {best_thresholds['max_draw_predictions']} at "
        f"{MAX_OVERRIDE_RATE_MULTIPLIER}x)"
    )
    print(
        "NOTE: tuned on the same matches used to fit the isotonic "
        "calibrator -- in-sample, optimistic. Worth re-validating "
        "out-of-sample as more results accumulate."
    )

    merged = align_models(calibrated, elo)

    ensemble = build_ensemble(merged, calibrated_columns, elo_columns, calibrator, rf_model, gb_model)

    ensemble = add_predictions(ensemble, override_thresholds)

    output = select_output_columns(ensemble)

    output.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 80)
    print("CALIBRATED ELO + xG/POISSON ENSEMBLE (draw override engine)")
    print("=" * 80)

    print()
    print(f"Base model weight: {BASE_WEIGHT:.0%}")
    print(f"Elo weight:        {ELO_WEIGHT:.0%}")
    print(f"Tree ensemble weight (RF+GB averaged): {TREE_WEIGHT:.0%}")
    print()
    print(f"Draw blend -- calibrated: {DRAW_CALIBRATED_WEIGHT:.0%}, tree: {DRAW_TREE_WEIGHT:.0%}")

    print()
    print("DRAW MODEL")
    print("-" * 80)
    print(f"Source:            {DRAW_CALIBRATOR_PATH.name} (isotonic regression)")
    print(f"Features from:     {STRENGTH_FEATURES_PATH.name}")
    print(f"Bounds:            [{CALIBRATOR_MIN_PROBABILITY:.0%}, {CALIBRATOR_MAX_PROBABILITY:.0%}]")

    print()
    print("PREDICTED RESULT DISTRIBUTION")
    print("-" * 80)
    print(output["ensemble_prediction"].value_counts())

    print()
    print("CONFIDENCE")
    print("-" * 80)
    print(output["ensemble_confidence"].value_counts())

    print()
    print("RECOMMENDATIONS")
    print("-" * 80)
    print(output["ensemble_recommendation"].value_counts())

    print()
    print("AVERAGE PROBABILITIES")
    print("-" * 80)
    print(f"HOME: {output['ensemble_home_win_prob'].mean():.2f}%")
    print(f"DRAW: {output['ensemble_draw_prob'].mean():.2f}%")
    print(f"AWAY: {output['ensemble_away_win_prob'].mean():.2f}%")

    print()
    print("DRAW ENGINE DIAGNOSTICS")
    print("-" * 80)
    print(f"Average draw signal:       {output['draw_signal'].mean():.2f}")
    print(f"Average equilibrium:       {output['draw_equilibrium'].mean():.3f}")
    print(f"Average low-score signal:  {output['draw_low_score_signal'].mean():.3f}")
    print(f"Average Poisson draw:      {output['draw_poisson_probability'].mean():.2f}%")
    print(f"Draw overrides applied:    {int(output['draw_override_applied'].sum())}")

    print()
    print("SAMPLE ENSEMBLE")
    print("-" * 80)
    print(output.head(20).to_string(index=False))

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()