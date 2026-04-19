#!/usr/bin/env python3
"""Filter unscorable problem IDs from Kaggle Nemotron train.csv.

Source: Kaggle discussion #689580 (hsiaosuan, 2026-04-09)
Discovery: Scorer regex `r'\\boxed\{([^}]*)(?:\}|$)'` stops at first `}`.
Problems with answers containing `}` NEVER score — wasted training signal.

Usage:
    python scripts/filter_unscorable.py --input train.csv --output train_filtered.csv
"""
import argparse
import pandas as pd

# Known unscorable problems (answers contain `}`)
UNSCORABLE_IDS = {
    "0d2e94ff",  # expected answer: `}}^`
    "0e375364",  # expected answer: `}/@`
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to train.csv")
    p.add_argument("--output", required=True, help="Path to filtered train.csv")
    p.add_argument("--id-col", default="id", help="Column name for problem ID")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    before = len(df)

    if args.id_col not in df.columns:
        print(f"WARNING: column '{args.id_col}' not found. Available: {list(df.columns)}")
        # Try common alternatives
        for col in ["id", "problem_id", "question_id"]:
            if col in df.columns:
                args.id_col = col
                break
        else:
            print("ERROR: no ID column found. Saving unchanged.")
            df.to_csv(args.output, index=False)
            return

    mask = ~df[args.id_col].isin(UNSCORABLE_IDS)
    filtered = df[mask].reset_index(drop=True)
    dropped = before - len(filtered)

    print(f"Input: {before} rows")
    print(f"Dropped (unscorable): {dropped} rows")
    print(f"Output: {len(filtered)} rows")
    print(f"Dropped IDs found: {set(df[args.id_col]).intersection(UNSCORABLE_IDS)}")

    filtered.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
