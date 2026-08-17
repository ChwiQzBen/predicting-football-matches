#!/usr/bin/env python3


from pathlib import Path


import pandas as pd


from normalizer import normalize_understat_matches




RAW_DIR = Path("data/raw/understat")
OUTPUT = Path("data/processed/epl_all_matches.csv")




def main():
    files = sorted(RAW_DIR.glob("EPL_*_matches.json"))


    if not files:
        raise FileNotFoundError("No EPL Understat files found.")


    frames = []


    for file in files:
        print(f"Loading {file.name}")


        df = normalize_understat_matches(file)


        # Season is encoded by the source filename.
        season = int(file.name.split("_")[1])
        df["season_start"] = season


        frames.append(df)


    combined = pd.concat(
        frames,
        ignore_index=True,
    )


    combined = (
        combined
        .sort_values(["date", "match_id"])
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    combined.to_csv(
        OUTPUT,
        index=False,
    )


    print()
    print(f"Total matches: {len(combined)}")
    print(f"Unique matches: {combined['match_id'].nunique()}")
    print(f"Teams: {len(set(combined['home_team']) | set(combined['away_team']))}")
    print(f"Date range: {combined['date'].min()} → {combined['date'].max()}")
    print(f"Saved: {OUTPUT}")




if __name__ == "__main__":
    main()
