#!/usr/bin/env python3
"""
edge_detection.py

Compares ensemble model probabilities (from ensemble_predictions_all.csv)
against bookmaker odds to flag +EV betting opportunities.

Two modes:

  single -- one match via CLI flags, odds typed in by hand:
    python edge_detection.py single --home "Arsenal" --away "Chelsea" \
        --odds-home 2.10 --odds-draw 3.40 --odds-away 3.20

    If the same two teams have played more than once in the ensemble
    file, add --date YYYY-MM-DD to disambiguate.

  batch -- many matches from a CSV of bookmaker odds:
    python edge_detection.py batch --odds-file data/processed/bookmaker_odds.csv

    The odds CSV needs: date, home_team, away_team, odds_home, odds_draw,
    odds_away. It's joined against ensemble_predictions_all.csv by
    date+home_team+away_team, the same match_key convention used
    throughout the rest of this project.

Both modes accept --bankroll (to show Kelly stake in currency, not just
as a fraction of bankroll) and --kelly-fraction (default 0.25, i.e.
quarter-Kelly -- full Kelly is mathematically "optimal" for long-run
growth but far more volatile than most people actually want to sit
through; quarter/half-Kelly is the standard practical compromise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

ENSEMBLE_PATH = DATA_DIR / "ensemble_predictions_all.csv"
OUTPUT_PATH = DATA_DIR / "edge_detection_results.csv"


# ============================================================
# CONFIG
# ============================================================

RESULT_LABELS = ["HOME", "DRAW", "AWAY"]

DEFAULT_KELLY_FRACTION = 0.25

# A bet is only flagged as +EV if it beats the raw break-even price
# (model_prob > 1/odds) -- see the note in calculate_edge_metrics()
# for why this is different from, and stricter than, "edge > 0".
MIN_EV_TO_FLAG = 0.0


# ============================================================
# HELPERS
# ============================================================

def build_match_key(df: pd.DataFrame) -> pd.Series:
    """
    Same convention used throughout the rest of this project
    (ensemble_predictions.py, ensemble_backtest.py, draw_calibration.py)
    so this script's joins line up with everything else.
    """

    dates = pd.to_datetime(df["date"], errors="coerce")
    home = df["home_team"].astype(str).str.strip()
    away = df["away_team"].astype(str).str.strip()

    return dates.dt.strftime("%Y-%m-%d %H:%M:%S") + "|" + home + "|" + away


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def load_ensemble_probabilities() -> pd.DataFrame:

    if not ENSEMBLE_PATH.exists():
        raise FileNotFoundError(
            f"Ensemble predictions not found:\n{ENSEMBLE_PATH}\n"
            "This script reads probabilities from that file -- it doesn't "
            "compute them itself. Run the ensemble pipeline first."
        )

    df = pd.read_csv(ENSEMBLE_PATH)

    required = [
        "date",
        "home_team",
        "away_team",
        "ensemble_home_win_prob",
        "ensemble_draw_prob",
        "ensemble_away_win_prob",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{ENSEMBLE_PATH.name} is missing required column(s): "
            f"{missing}\nFound columns: {list(df.columns)}"
        )

    df = df.copy()

    prob_columns = [
        "ensemble_home_win_prob",
        "ensemble_draw_prob",
        "ensemble_away_win_prob",
    ]

    values = df[prob_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Ensemble files in this project store these as percentages
    # (0-100), but detect and handle a 0-1 fraction file gracefully
    # too, same defensive pattern used elsewhere in the pipeline.
    if np.nanmean(values) > 1.0:
        values = values / 100.0

    df[prob_columns] = values

    df["match_key"] = build_match_key(df)

    return df


def lookup_match(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    date: str | None,
) -> pd.Series:

    home_mask = df["home_team"].astype(str).str.strip().str.lower() == home_team.strip().lower()
    away_mask = df["away_team"].astype(str).str.strip().str.lower() == away_team.strip().lower()

    candidates = df[home_mask & away_mask]

    if date is not None:
        parsed_date = pd.to_datetime(date, errors="coerce")
        if pd.isna(parsed_date):
            raise ValueError(f"Could not parse --date value: {date!r} (expected YYYY-MM-DD)")
        candidates = candidates[
            pd.to_datetime(candidates["date"], errors="coerce").dt.date == parsed_date.date()
        ]

    if len(candidates) == 0:

        # Help the user spot a typo rather than just failing silently.
        near_home = df[df["home_team"].astype(str).str.contains(home_team, case=False, na=False)]
        near_away = df[df["away_team"].astype(str).str.contains(away_team, case=False, na=False)]

        hint = ""
        if not near_home.empty or not near_away.empty:
            sample_fixtures = pd.concat([near_home, near_away]).drop_duplicates()
            sample_lines = "\n".join(
                f"  {row['date']} | {row['home_team']} vs {row['away_team']}"
                for _, row in sample_fixtures.head(5).iterrows()
            )
            hint = f"\n\nSimilar fixtures found in {ENSEMBLE_PATH.name}:\n{sample_lines}"

        raise ValueError(
            f"No match found for '{home_team}' vs '{away_team}'"
            f"{' on ' + date if date else ''} in {ENSEMBLE_PATH.name}.{hint}"
        )

    if len(candidates) > 1:
        fixture_lines = "\n".join(
            f"  {row['date']} | {row['home_team']} vs {row['away_team']}"
            for _, row in candidates.iterrows()
        )
        raise ValueError(
            f"Multiple fixtures found for '{home_team}' vs '{away_team}'. "
            f"Add --date to disambiguate:\n{fixture_lines}"
        )

    return candidates.iloc[0]


# ============================================================
# VIG REMOVAL
# ============================================================

def remove_vig(odds_home: float, odds_draw: float, odds_away: float) -> tuple[dict[str, float], float]:
    """
    Converts raw decimal bookmaker odds into vig-free ("fair")
    probabilities using the standard proportional method: implied
    probabilities (1/odds) are normalised to sum to 1.

    This is the simplest, most widely used overround-removal method.
    It doesn't correct for favourite-longshot bias the way Shin's
    method does -- worth knowing if you want to compare against a
    more sophisticated method later, but this is a reasonable,
    transparent baseline.

    Returns (fair_probabilities_dict, overround). overround > 1.0
    is the bookmaker's margin; overround <= 1.0 would mean the odds
    imply zero or negative margin, which is a strong signal of a
    data-entry mistake worth double-checking.
    """

    for label, odds in zip(RESULT_LABELS, [odds_home, odds_draw, odds_away]):
        if odds <= 1.0:
            raise ValueError(
                f"{label} odds must be greater than 1.0 (got {odds}). "
                "Decimal odds of 1.0 or less imply zero or negative payout."
            )

    implied = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away])
    overround = float(implied.sum())

    fair_values = implied / overround

    fair = dict(zip(RESULT_LABELS, fair_values))

    return fair, overround


# ============================================================
# EDGE / EV / KELLY
# ============================================================

def calculate_edge_metrics(
    model_prob: float,
    fair_prob: float,
    decimal_odds: float,
    bankroll: float | None,
    kelly_fraction: float,
) -> dict:
    """
    edge:  model_prob - fair_prob (vig-removed). Tells you whether
           your model disagrees with the bookmaker's true assessment
           of the match.

    ev:    model_prob * decimal_odds - 1. Tells you whether you'd
           actually turn a profit betting at THIS price, including
           the bookmaker's margin. This is the number that should
           drive the bet/no-bet decision -- it's stricter than
           edge > 0, since fair_prob < 1/decimal_odds whenever the
           book has a margin. A match can show positive edge (you
           beat the bookmaker's true view) while still being -EV
           (you don't beat their actual price). Both are reported;
           the flag uses EV.
    """

    edge = model_prob - fair_prob

    ev = model_prob * decimal_odds - 1.0

    # Kelly criterion: f* = (b*p - q) / b, b = net decimal odds (payout
    # per unit staked, excluding the stake itself), q = 1 - p.
    b = decimal_odds - 1.0
    q = 1.0 - model_prob

    kelly_full = (b * model_prob - q) / b if b > 0 else 0.0
    kelly_full = max(kelly_full, 0.0)

    kelly_stake_fraction = kelly_full * kelly_fraction
    kelly_stake_amount = kelly_stake_fraction * bankroll if bankroll is not None else None

    return {
        "model_prob": model_prob,
        "fair_prob": fair_prob,
        "decimal_odds": decimal_odds,
        "edge": edge,
        "ev": ev,
        "kelly_full_fraction": kelly_full,
        "kelly_stake_fraction": kelly_stake_fraction,
        "kelly_stake_amount": kelly_stake_amount,
        "has_value": ev > MIN_EV_TO_FLAG,
    }


def analyse_match(
    model_probs: dict[str, float],
    odds: dict[str, float],
    bankroll: float | None,
    kelly_fraction: float,
) -> tuple[dict[str, dict], float]:

    fair, overround = remove_vig(odds["HOME"], odds["DRAW"], odds["AWAY"])

    results = {}

    for label in RESULT_LABELS:
        results[label] = calculate_edge_metrics(
            model_prob=model_probs[label],
            fair_prob=fair[label],
            decimal_odds=odds[label],
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
        )

    return results, overround


# ============================================================
# SINGLE MATCH DISPLAY
# ============================================================

def print_single_match_report(
    home_team: str,
    away_team: str,
    match_date,
    results: dict[str, dict],
    overround: float,
    bankroll: float | None,
    kelly_fraction: float,
):

    print()
    print("=" * 80)
    print(f"{home_team} vs {away_team}  ({match_date})")
    print("=" * 80)

    print()
    print(f"Bookmaker overround: {overround:.4f}  (margin: {(overround - 1.0):.2%})")

    if overround <= 1.0:
        print(
            "WARNING: overround <= 1.0 -- these odds imply zero or negative "
            "bookmaker margin, which is very unusual. Double-check the odds "
            "were entered correctly."
        )

    print()
    header = (
        f"{'Outcome':<7}{'Model':>9}{'Fair':>9}{'Odds':>8}"
        f"{'Edge':>9}{'EV':>9}{'Kelly%':>9}{'Flag':>10}"
    )
    print(header)
    print("-" * len(header))

    for label in RESULT_LABELS:

        row = results[label]

        flag = "+EV" if row["has_value"] else "no value"

        print(
            f"{label:<7}"
            f"{row['model_prob']:>8.1%} "
            f"{row['fair_prob']:>8.1%} "
            f"{row['decimal_odds']:>7.2f} "
            f"{row['edge']:>+8.1%} "
            f"{row['ev']:>+8.1%} "
            f"{row['kelly_stake_fraction']:>8.2%} "
            f"{flag:>9}"
        )

    print()

    value_bets = [label for label in RESULT_LABELS if results[label]["has_value"]]

    if not value_bets:
        print("No +EV opportunities found at these odds.")
    else:
        print("+EV OPPORTUNITIES")
        print("-" * 80)
        for label in value_bets:
            row = results[label]
            stake_line = f"{row['kelly_stake_fraction']:.2%} of bankroll"
            if bankroll is not None:
                stake_line += f"  ({row['kelly_stake_amount']:.2f})"
            print(
                f"  {label}: edge {row['edge']:+.1%}, EV {row['ev']:+.1%}, "
                f"stake (quarter-Kelly x{kelly_fraction}): {stake_line}"
            )

    print()


# ============================================================
# SINGLE MODE
# ============================================================

def run_single(args: argparse.Namespace):

    ensemble = load_ensemble_probabilities()

    match_row = lookup_match(ensemble, args.home, args.away, args.date)

    model_probs = {
        "HOME": float(match_row["ensemble_home_win_prob"]),
        "DRAW": float(match_row["ensemble_draw_prob"]),
        "AWAY": float(match_row["ensemble_away_win_prob"]),
    }

    odds = {
        "HOME": args.odds_home,
        "DRAW": args.odds_draw,
        "AWAY": args.odds_away,
    }

    results, overround = analyse_match(
        model_probs=model_probs,
        odds=odds,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
    )

    print_single_match_report(
        home_team=match_row["home_team"],
        away_team=match_row["away_team"],
        match_date=match_row["date"],
        results=results,
        overround=overround,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
    )


# ============================================================
# BATCH MODE
# ============================================================

def load_odds_file(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"Odds file not found:\n{path}")

    odds = pd.read_csv(path)

    date_column = find_column(odds, ["date", "match_date", "Date"])
    home_column = find_column(odds, ["home_team", "home", "Home", "HomeTeam"])
    away_column = find_column(odds, ["away_team", "away", "Away", "AwayTeam"])
    odds_home_column = find_column(odds, ["odds_home", "home_odds", "odds_h", "B365H"])
    odds_draw_column = find_column(odds, ["odds_draw", "draw_odds", "odds_d", "B365D"])
    odds_away_column = find_column(odds, ["odds_away", "away_odds", "odds_a", "B365A"])

    required_map = {
        "date": date_column,
        "home_team": home_column,
        "away_team": away_column,
        "odds_home": odds_home_column,
        "odds_draw": odds_draw_column,
        "odds_away": odds_away_column,
    }

    missing = [name for name, column in required_map.items() if column is None]

    if missing:
        raise ValueError(
            f"Could not find column(s) for {missing} in {path.name}.\n"
            f"Found columns: {list(odds.columns)}\n"
            "Expected at least: date, home_team, away_team, odds_home, "
            "odds_draw, odds_away (or common aliases like B365H/B365D/B365A)."
        )

    odds = odds.rename(
        columns={
            date_column: "date",
            home_column: "home_team",
            away_column: "away_team",
            odds_home_column: "odds_home",
            odds_draw_column: "odds_draw",
            odds_away_column: "odds_away",
        }
    )

    odds["match_key"] = build_match_key(odds)

    return odds


def run_batch(args: argparse.Namespace):

    ensemble = load_ensemble_probabilities()
    odds_df = load_odds_file(Path(args.odds_file))

    merged = odds_df.merge(
        ensemble,
        on="match_key",
        how="left",
        suffixes=("", "_ensemble"),
    )

    unmatched = merged["ensemble_home_win_prob"].isna()

    if unmatched.any():
        unmatched_fixtures = merged.loc[
            unmatched, ["date", "home_team", "away_team"]
        ]
        print(
            f"WARNING: {int(unmatched.sum())} of {len(merged)} fixtures in "
            f"{Path(args.odds_file).name} had no match in {ENSEMBLE_PATH.name} "
            "and will be skipped:"
        )
        for _, row in unmatched_fixtures.iterrows():
            print(f"  {row['date']} | {row['home_team']} vs {row['away_team']}")
        print()

    matched = merged.loc[~unmatched].copy()

    if matched.empty:
        raise ValueError(
            "No fixtures in the odds file matched any row in "
            f"{ENSEMBLE_PATH.name}. Check date/team-name alignment between "
            "the two files."
        )

    records = []

    for _, row in matched.iterrows():

        model_probs = {
            "HOME": float(row["ensemble_home_win_prob"]),
            "DRAW": float(row["ensemble_draw_prob"]),
            "AWAY": float(row["ensemble_away_win_prob"]),
        }

        odds = {
            "HOME": float(row["odds_home"]),
            "DRAW": float(row["odds_draw"]),
            "AWAY": float(row["odds_away"]),
        }

        try:
            results, overround = analyse_match(
                model_probs=model_probs,
                odds=odds,
                bankroll=args.bankroll,
                kelly_fraction=args.kelly_fraction,
            )
        except ValueError as error:
            print(
                f"SKIPPED {row['home_team']} vs {row['away_team']} "
                f"({row['date']}): {error}"
            )
            continue

        for label in RESULT_LABELS:
            metrics = results[label]
            records.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "outcome": label,
                    "model_prob": metrics["model_prob"],
                    "fair_prob": metrics["fair_prob"],
                    "decimal_odds": metrics["decimal_odds"],
                    "overround": overround,
                    "edge": metrics["edge"],
                    "ev": metrics["ev"],
                    "kelly_full_fraction": metrics["kelly_full_fraction"],
                    "kelly_stake_fraction": metrics["kelly_stake_fraction"],
                    "kelly_stake_amount": metrics["kelly_stake_amount"],
                    "has_value": metrics["has_value"],
                }
            )

    output = pd.DataFrame(records)

    output.to_csv(OUTPUT_PATH, index=False)

    print_batch_summary(output, args.bankroll, args.kelly_fraction)

    print(f"Saved: {OUTPUT_PATH}")


def print_batch_summary(
    output: pd.DataFrame,
    bankroll: float | None,
    kelly_fraction: float,
):

    n_matches = output["date"].nunique() if "date" in output.columns else 0
    n_rows = len(output)

    value_bets = output[output["has_value"]]

    print()
    print("=" * 80)
    print("BATCH EDGE DETECTION SUMMARY")
    print("=" * 80)

    print()
    print(f"Matches analysed:        {n_matches}")
    print(f"Outcome rows evaluated:  {n_rows}  (3 per match: HOME/DRAW/AWAY)")
    print(f"+EV opportunities found: {len(value_bets)}")

    if len(value_bets) == 0:
        print()
        print("No +EV opportunities found at the odds provided.")
        return

    print()
    print("+EV BREAKDOWN BY OUTCOME")
    print("-" * 80)
    print(value_bets["outcome"].value_counts().reindex(RESULT_LABELS, fill_value=0))

    print()
    print(f"Average edge among +EV picks: {value_bets['edge'].mean():+.2%}")
    print(f"Average EV among +EV picks:   {value_bets['ev'].mean():+.2%}")

    if bankroll is not None:
        total_stake = value_bets["kelly_stake_amount"].sum()
        print(
            f"Total recommended stake (quarter-Kelly x{kelly_fraction}, "
            f"bankroll {bankroll:.2f}): {total_stake:.2f} "
            f"({total_stake / bankroll:.2%} of bankroll)"
        )
    else:
        total_stake_fraction = value_bets["kelly_stake_fraction"].sum()
        print(
            f"Total recommended stake (quarter-Kelly x{kelly_fraction}): "
            f"{total_stake_fraction:.2%} of bankroll  "
            "(pass --bankroll to see this in currency)"
        )

    print()
    print("TOP +EV OPPORTUNITIES (by EV)")
    print("-" * 80)

    top = value_bets.sort_values("ev", ascending=False).head(10)

    for _, row in top.iterrows():
        print(
            f"  {row['date']} | {row['home_team']} vs {row['away_team']} "
            f"| {row['outcome']:<5} | edge {row['edge']:+.1%} | "
            f"EV {row['ev']:+.1%} | odds {row['decimal_odds']:.2f} | "
            f"Kelly stake {row['kelly_stake_fraction']:.2%}"
        )


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Compare ensemble probabilities against bookmaker odds to flag +EV bets."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Analyse one match via CLI flags")
    single.add_argument("--home", required=True, help="Home team name, as it appears in ensemble_predictions_all.csv")
    single.add_argument("--away", required=True, help="Away team name, as it appears in ensemble_predictions_all.csv")
    single.add_argument("--date", default=None, help="YYYY-MM-DD, only needed if the team pair has multiple fixtures")
    single.add_argument("--odds-home", type=float, required=True, help="Decimal odds for a HOME win")
    single.add_argument("--odds-draw", type=float, required=True, help="Decimal odds for a DRAW")
    single.add_argument("--odds-away", type=float, required=True, help="Decimal odds for an AWAY win")
    single.add_argument("--bankroll", type=float, default=None, help="Optional: show Kelly stake in currency, not just as a fraction")
    single.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION, help=f"Fraction of full Kelly to recommend (default {DEFAULT_KELLY_FRACTION} = quarter-Kelly)")

    batch = subparsers.add_parser("batch", help="Analyse many matches from a CSV of bookmaker odds")
    batch.add_argument("--odds-file", required=True, help="CSV with date, home_team, away_team, odds_home, odds_draw, odds_away")
    batch.add_argument("--bankroll", type=float, default=None, help="Optional: show Kelly stakes in currency, not just as a fraction")
    batch.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION, help=f"Fraction of full Kelly to recommend (default {DEFAULT_KELLY_FRACTION} = quarter-Kelly)")

    return parser


def main():

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.mode == "single":
            run_single(args)
        elif args.mode == "batch":
            run_batch(args)
        else:
            parser.print_help()
            sys.exit(1)
    except (FileNotFoundError, ValueError) as error:
        # These are expected, user-actionable errors (bad odds, team not
        # found, missing file, etc.) -- print just the message, not a
        # full traceback, and exit non-zero so scripting/CI can detect
        # failure.
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
