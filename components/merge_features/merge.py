import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length",    type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf",     type=str, required=True)
    parser.add_argument("--sbert",     type=str, required=True)
    parser.add_argument("--out",       type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    keys = ["asin", "reviewerID"]

    length_df    = pd.read_parquet(args.length)
    sentiment_df = pd.read_parquet(args.sentiment)
    tfidf_df     = pd.read_parquet(args.tfidf)
    sbert_df     = pd.read_parquet(args.sbert)

    merged = (
        length_df
        .merge(sentiment_df, on=keys, how="inner")
        .merge(tfidf_df,     on=keys, how="inner")
        .merge(sbert_df,     on=keys, how="inner")
    )

    if "overall" not in merged.columns:
        raise RuntimeError("'overall' column missing after merge. Check length.py.")

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Merged shape:", merged.shape)
    print("Columns:", merged.columns.tolist())


if __name__ == "__main__":
    main()
