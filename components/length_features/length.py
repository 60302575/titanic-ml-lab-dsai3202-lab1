import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out",  type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)

    df["review_length_words"] = df["reviewText"].str.split().str.len()
    df["review_length_chars"] = df["reviewText"].str.len()

    cols = ["asin", "reviewerID", "overall", "review_length_words", "review_length_chars"]
    cols = [c for c in cols if c in df.columns]
    out_df = df[cols]

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Length features computed:", len(out_df))


if __name__ == "__main__":
    main()
