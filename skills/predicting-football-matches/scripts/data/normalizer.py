#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def normalize_understat_matches(
    input_file: str | Path,
) -> pd.DataFrame:
    input_file = Path(input_file)

    with input_file.open("r", encoding="utf-8") as f:
        matches = json.load(f)

    rows = []

    for match in matches:
        rows.append({
            "match_id": int(match["id"]),
            "date": pd.to_datetime(match["datetime"]),
            "home_team": match["h"]["title"],
            "away_team": match["a"]["title"],
            "home_goals": int(match["goals"]["h"]),
            "away_goals": int(match["goals"]["a"]),
            "home_xg": float(match["xG"]["h"]),
            "away_xg": float(match["xG"]["a"]),
            "home_win_prob": float(match["forecast"]["w"]),
            "draw_prob": float(match["forecast"]["d"]),
            "away_win_prob": float(match["forecast"]["l"]),
        })

    df = pd.DataFrame(rows)

    return df.sort_values("date").reset_index(drop=True)


def save_normalized_data(
    df: pd.DataFrame,
    output_file: str | Path,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
    )


if __name__ == "__main__":
    source = Path(
        "data/raw/understat/EPL_2025_matches.json"
    )

    output = Path(
        "data/processed/epl_2025_matches.csv"
    )

    df = normalize_understat_matches(source)

    save_normalized_data(df, output)

    print(f"Matches normalized: {len(df)}")
    print(f"Saved: {output}")
    print()
    print(df.head().to_string(index=False))
