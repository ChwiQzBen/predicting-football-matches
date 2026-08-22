#!/usr/bin/env python3
"""
rf_gb_1x2_classifier.py

Standalone Random Forest and Gradient Boosting 1X2 classifiers.
Trained on epl_strength_features.csv (pre-cutoff matches, same
population draw_calibration.py trains on) and evaluated on the
EXACT SAME matches already scored in ensemble_backtest.csv, so
results are directly, honestly comparable to the current ensemble.

This does NOT modify or feed into ensemble_predictions.py. It's a
standalone check: does RF/GB actually beat the current ensemble
before any integration work happens.

Usage:
    python rf_gb_1x2_classifier.py
"""

from __future__ import annotations

import pickle
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

from draw_calibration import (  # noqa: E402
    CALIBRATION_CUTOFF,
    calculate_equilibrium,
    calculate_low_score_signal,
    find_probability_columns,
    find_strength_columns,
    find_xg_columns,
    get_actual_result,
    poisson_draw_probability,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

SOURCE_PATH = DATA_DIR / "epl_strength_features.csv"
ENSEMBLE_BACKTEST_PATH = DATA_DIR / "ensemble_backtest.csv"

COMPARISON_OUTPUT_PATH = DATA_DIR / "rf_gb_comparison.csv"
RF_MODEL_PATH = DATA_DIR / "model_1x2_random_forest.pkl"
GB_MODEL_PATH = DATA_DIR / "model_1x2_gradient_boosting.pkl"


# ============================================================
# CONFIG
# ============================================================

RESULT_ORDER = ["HOME", "DRAW", "AWAY"]

# Deliberately conservative for ~1000-1800 training rows -- deep
# trees / many estimators on a dataset this size overfit fast, and
# the whole point of this comparison is an honest read, not a
# flattering one.
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)

GB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_samples_leaf=20,
    random_state=42,
)


# ============================================================
# HELPERS
# ============================================================

def build_match_key(df: pd.DataFrame) -> pd.Series:

    dates = pd.to_datetime(df["date"], errors="coerce")
    home = df["home_team"].astype(str).str.strip()
    away = df["away_team"].astype(str).str.strip()

    return dates.dt.strftime("%Y-%m-%d %H:%M:%S") + "|" + home + "|" + away


# ============================================================
# FEATURE ENGINEERING
#
# Reuses the same domain functions the draw engine already relies
# on (equilibrium, low-score signal, Poisson draw probability), so
# RF/GB get access to the same informed signals as the hand-tuned
# draw_signal formula, rather than starting from raw columns alone.
# Whatever these models learn is a genuine alternative to the
# hand-tuned weights, not a comparison against a worse feature set.
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    prob_columns = find_probability_columns(df)

    base_home = pd.to_numeric(df[prob_columns[0]], errors="coerce").to_numpy(dtype=float)
    base_draw = pd.to_numeric(df[prob_columns[1]], errors="coerce").to_numpy(dtype=float)
    base_away = pd.to_numeric(df[prob_columns[2]], errors="coerce").to_numpy(dtype=float)

    if np.nanmean([base_home, base_draw, base_away]) > 1.0:
        base_home, base_draw, base_away = base_home / 100.0, base_draw / 100.0, base_away / 100.0

    home_strength_column, away_strength_column = find_strength_columns(df)

    if home_strength_column is not None and away_strength_column is not None:
        home_strength = pd.to_numeric(df[home_strength_column], errors="coerce").to_numpy(dtype=float)
        away_strength = pd.to_numeric(df[away_strength_column], errors="coerce").to_numpy(dtype=float)
    else:
        home_strength = base_home
        away_strength = base_away

    home_xg_column, away_xg_column = find_xg_columns(df)

    if home_xg_column is not None and away_xg_column is not None:
        home_xg = pd.to_numeric(df[home_xg_column], errors="coerce").to_numpy(dtype=float)
        away_xg = pd.to_numeric(df[away_xg_column], errors="coerce").to_numpy(dtype=float)
    else:
        home_xg = 0.8 + 1.8 * base_home
        away_xg = 0.7 + 1.6 * base_away

    equilibrium = calculate_equilibrium(home_strength, away_strength)
    low_score = calculate_low_score_signal(home_xg, away_xg)
    poisson_draw = poisson_draw_probability(home_xg, away_xg)

    features = pd.DataFrame(
        {
            "base_home_prob": base_home,
            "base_draw_prob": base_draw,
            "base_away_prob": base_away,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "xg_diff": home_xg - away_xg,
            "total_xg": home_xg + away_xg,
            "home_strength": home_strength,
            "away_strength": away_strength,
            "strength_diff": home_strength - away_strength,
            "equilibrium": equilibrium,
            "low_score_signal": low_score,
            "poisson_draw_prob": poisson_draw,
        },
        index=df.index,
    )

    return features


FEATURE_COLUMNS = [
    "base_home_prob",
    "base_draw_prob",
    "base_away_prob",
    "home_xg",
    "away_xg",
    "xg_diff",
    "total_xg",
    "home_strength",
    "away_strength",
    "strength_diff",
    "equilibrium",
    "low_score_signal",
    "poisson_draw_prob",
]


# ============================================================
# DATA LOADING / SPLITTING
# ============================================================

def load_and_split_data():

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Historical match data not found:\n{SOURCE_PATH}")

    if not ENSEMBLE_BACKTEST_PATH.exists():
        raise FileNotFoundError(
            f"Ensemble backtest not found:\n{ENSEMBLE_BACKTEST_PATH}\n"
            "Run ensemble_backtest.py first -- this script evaluates RF/GB "
            "on the exact same matches already scored there, so the "
            "comparison is apples-to-apples."
        )

    source = pd.read_csv(SOURCE_PATH)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["match_key"] = build_match_key(source)

    ensemble_backtest = pd.read_csv(ENSEMBLE_BACKTEST_PATH)
    ensemble_backtest["date"] = pd.to_datetime(ensemble_backtest["date"], errors="coerce")

    if "match_key" not in ensemble_backtest.columns:
        ensemble_backtest["match_key"] = build_match_key(ensemble_backtest)

    eval_keys = set(ensemble_backtest["match_key"])

    actual_result = get_actual_result(source)
    source["actual_result"] = actual_result

    features = build_features(source)

    full = pd.concat([source[["date", "home_team", "away_team", "match_key", "actual_result"]], features], axis=1)
    full = full.dropna(subset=["actual_result", *FEATURE_COLUMNS])

    train = full[full["date"] < CALIBRATION_CUTOFF].copy()
    test = full[full["match_key"].isin(eval_keys)].copy()

    if len(train) < 300:
        raise ValueError(
            f"Only {len(train)} pre-cutoff training rows available after "
            "dropping missing values -- too few to train RF/GB reliably."
        )

    if test.empty:
        raise ValueError(
            "None of the matches in ensemble_backtest.csv could be matched "
            "back to epl_strength_features.csv by date/home_team/away_team. "
            "Check that both files describe the same matches."
        )

    print(f"Training matches (pre-cutoff): {len(train)}")
    print(f"Evaluation matches (matched to ensemble_backtest.csv): {len(test)} / {len(ensemble_backtest)}")

    return train, test, ensemble_backtest


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def calculate_brier_score_3class(actual: np.ndarray, probabilities: dict[str, np.ndarray]) -> float:
    """
    probabilities: dict RESULT_ORDER label -> np.ndarray of probabilities
    for that outcome, already aligned to `actual`.
    """

    squared_error = np.zeros(len(actual), dtype=float)

    for label in RESULT_ORDER:
        indicator = (actual == label).astype(float)
        squared_error += (probabilities[label] - indicator) ** 2

    return float(squared_error.mean() / 3.0)


def train_and_evaluate(
    model,
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:

    X_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["actual_result"].to_numpy()

    X_test = test[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test["actual_result"].to_numpy()

    sample_weight = compute_sample_weight("balanced", y_train)

    model.fit(X_train, y_train, sample_weight=sample_weight)

    # --------------------------------------------------------
    # IMPORTANT: never assume class column order. predict_proba()'s
    # columns correspond exactly to model.classes_, in whatever order
    # sklearn assigned internally (typically alphabetical for string
    # labels: AWAY, DRAW, HOME) -- NOT necessarily RESULT_ORDER. This
    # is the same class of bug already found and fixed once in
    # ensemble_backtest.py's log_loss call. Using model.classes_
    # directly here sidesteps it entirely rather than repeating it.
    # --------------------------------------------------------

    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)

    test_probabilities_raw = model.predict_proba(X_test)
    class_order = list(model.classes_)

    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)

    test_logloss = log_loss(y_test, test_probabilities_raw, labels=class_order)

    probabilities_by_label = {
        label: test_probabilities_raw[:, class_order.index(label)] for label in RESULT_ORDER
    }

    test_brier = calculate_brier_score_3class(y_test, probabilities_by_label)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_predictions, labels=RESULT_ORDER, zero_division=0
    )

    matrix = confusion_matrix(y_test, test_predictions, labels=RESULT_ORDER)

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))

    return {
        "model_name": model_name,
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "overfit_gap": train_accuracy - test_accuracy,
        "test_logloss": test_logloss,
        "test_brier": test_brier,
        "precision": dict(zip(RESULT_ORDER, precision)),
        "recall": dict(zip(RESULT_ORDER, recall)),
        "f1": dict(zip(RESULT_ORDER, f1)),
        "support": dict(zip(RESULT_ORDER, support)),
        "confusion_matrix": matrix,
        "feature_importances": importances,
        "test_predictions": test_predictions,
        "probabilities_by_label": probabilities_by_label,
        "test_index": test.index,
    }


# ============================================================
# ENSEMBLE BASELINE (recomputed on the identical match set)
# ============================================================

def evaluate_ensemble_baseline(test: pd.DataFrame, ensemble_backtest: pd.DataFrame) -> dict:

    matched = ensemble_backtest[ensemble_backtest["match_key"].isin(test["match_key"])].copy()

    actual = matched["actual_result"].to_numpy()
    predicted = matched["predicted_result"].to_numpy()

    accuracy = accuracy_score(actual, predicted)

    probabilities_by_label = {
        "HOME": matched["ensemble_home_win_prob"].to_numpy(dtype=float),
        "DRAW": matched["ensemble_draw_prob"].to_numpy(dtype=float),
        "AWAY": matched["ensemble_away_win_prob"].to_numpy(dtype=float),
    }

    # ensemble_backtest.csv stores these as 0-1 fractions already
    # (ensemble_backtest.py's own output columns), not percentages --
    # confirm and normalise defensively either way.
    stacked = np.column_stack([probabilities_by_label[label] for label in RESULT_ORDER])
    if np.nanmean(stacked) > 1.0:
        for label in RESULT_ORDER:
            probabilities_by_label[label] = probabilities_by_label[label] / 100.0

    brier = calculate_brier_score_3class(actual, probabilities_by_label)

    # Same alphabetical-column-order fix as ensemble_backtest.py's own
    # log_loss call -- reorder columns to match sorted(RESULT_ORDER)
    # before calling log_loss, since sklearn ignores the `labels=`
    # ordering for column correspondence.
    alphabetical_order = sorted(RESULT_ORDER)
    reordered = np.column_stack([probabilities_by_label[label] for label in alphabetical_order])
    logloss = log_loss(actual, reordered, labels=alphabetical_order)

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=RESULT_ORDER, zero_division=0
    )

    matrix = confusion_matrix(actual, predicted, labels=RESULT_ORDER)

    return {
        "model_name": "Current Ensemble",
        "test_accuracy": accuracy,
        "test_logloss": logloss,
        "test_brier": brier,
        "precision": dict(zip(RESULT_ORDER, precision)),
        "recall": dict(zip(RESULT_ORDER, recall)),
        "f1": dict(zip(RESULT_ORDER, f1)),
        "support": dict(zip(RESULT_ORDER, support)),
        "confusion_matrix": matrix,
        "n_matches": len(matched),
    }


# ============================================================
# REPORTING
# ============================================================

def print_comparison_table(results_list: list[dict]):

    print()
    print("=" * 80)
    print("MODEL COMPARISON (identical evaluation match set)")
    print("=" * 80)

    print()
    header = f"{'Model':<20}{'Accuracy':>12}{'Log Loss':>12}{'Brier':>12}{'Overfit gap':>14}"
    print(header)
    print("-" * len(header))

    for result in results_list:
        gap = result.get("overfit_gap")
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        print(
            f"{result['model_name']:<20}"
            f"{result['test_accuracy']:>12.4f}"
            f"{result['test_logloss']:>12.4f}"
            f"{result['test_brier']:>12.4f}"
            f"{gap_str:>14}"
        )

    print()
    print("DRAW CLASS: precision / recall / F1")
    print("-" * 80)
    print(f"{'Model':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    for result in results_list:
        print(
            f"{result['model_name']:<20}"
            f"{result['precision']['DRAW']:>12.4f}"
            f"{result['recall']['DRAW']:>12.4f}"
            f"{result['f1']['DRAW']:>12.4f}"
        )

    best_accuracy = max(results_list, key=lambda r: r["test_accuracy"])
    best_logloss = min(results_list, key=lambda r: r["test_logloss"])

    print()
    print(f"Best accuracy:  {best_accuracy['model_name']} ({best_accuracy['test_accuracy']:.4f})")
    print(f"Best log loss:  {best_logloss['model_name']} ({best_logloss['test_logloss']:.4f})")


def print_feature_importances(result: dict):

    print()
    print(f"{result['model_name']} FEATURE IMPORTANCES")
    print("-" * 80)

    sorted_importances = sorted(
        result["feature_importances"].items(), key=lambda item: item[1], reverse=True
    )

    for feature, importance in sorted_importances:
        bar = "#" * int(importance * 50)
        print(f"  {feature:<20} {importance:.4f}  {bar}")


def print_confusion_matrix(result: dict):

    print()
    print(f"{result['model_name']} CONFUSION MATRIX (rows=actual, cols=predicted)")
    print("-" * 80)

    header = " " * 8 + "".join(f"{label:>8}" for label in RESULT_ORDER)
    print(header)

    for label, row in zip(RESULT_ORDER, result["confusion_matrix"]):
        row_str = "".join(f"{value:>8}" for value in row)
        print(f"{label:<8}{row_str}")


# ============================================================
# SAVE
# ============================================================

def save_models(rf_result: dict, gb_result: dict):

    with open(RF_MODEL_PATH, "wb") as file:
        pickle.dump(rf_result["model"], file)

    with open(GB_MODEL_PATH, "wb") as file:
        pickle.dump(gb_result["model"], file)

    print()
    print(f"Saved: {RF_MODEL_PATH}")
    print(f"Saved: {GB_MODEL_PATH}")


def save_comparison_csv(test: pd.DataFrame, rf_result: dict, gb_result: dict):

    output = test[["date", "home_team", "away_team", "actual_result"]].copy()

    for label in RESULT_ORDER:
        output[f"rf_{label.lower()}_prob"] = rf_result["probabilities_by_label"][label]
        output[f"gb_{label.lower()}_prob"] = gb_result["probabilities_by_label"][label]

    output["rf_prediction"] = rf_result["test_predictions"]
    output["gb_prediction"] = gb_result["test_predictions"]
    output["rf_correct"] = output["rf_prediction"] == output["actual_result"]
    output["gb_correct"] = output["gb_prediction"] == output["actual_result"]

    output.to_csv(COMPARISON_OUTPUT_PATH, index=False)

    print(f"Saved: {COMPARISON_OUTPUT_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():

    train, test, ensemble_backtest = load_and_split_data()

    print()
    print("Training Random Forest...")
    rf_result = train_and_evaluate(
        RandomForestClassifier(**RF_PARAMS), "Random Forest", train, test
    )

    print("Training Gradient Boosting...")
    gb_result = train_and_evaluate(
        GradientBoostingClassifier(**GB_PARAMS), "Gradient Boosting", train, test
    )

    print("Evaluating current ensemble on the same match set...")
    ensemble_result = evaluate_ensemble_baseline(test, ensemble_backtest)

    results_list = [ensemble_result, rf_result, gb_result]

    print_comparison_table(results_list)

    print_feature_importances(rf_result)
    print_feature_importances(gb_result)

    print_confusion_matrix(rf_result)
    print_confusion_matrix(gb_result)
    print_confusion_matrix(ensemble_result)

    save_models(rf_result, gb_result)
    save_comparison_csv(test, rf_result, gb_result)

    print()
    print("=" * 80)
    print("NOTE ON OVERFITTING")
    print("=" * 80)
    print(
        f"RF train accuracy {rf_result['train_accuracy']:.4f} vs test "
        f"accuracy {rf_result['test_accuracy']:.4f} "
        f"(gap {rf_result['overfit_gap']:+.4f}). A large positive gap means "
        "the model memorised training data rather than learning a "
        "generalisable pattern -- treat its test numbers with more "
        "suspicion the bigger this gap is."
    )
    print(
        f"GB train accuracy {gb_result['train_accuracy']:.4f} vs test "
        f"accuracy {gb_result['test_accuracy']:.4f} "
        f"(gap {gb_result['overfit_gap']:+.4f})."
    )

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
