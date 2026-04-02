import argparse
import os
import pandas as pd
import nltk

# Download to a writable local path
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)
nltk.download("vader_lexicon", download_dir=nltk_data_dir, quiet=False)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out",  type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    sia = SentimentIntensityAnalyzer()
    scores = df["reviewText"].fillna("").apply(sia.polarity_scores)
    df["sentiment_pos"]      = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"]      = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"]      = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])
    cols = ["asin", "reviewerID", "overall", "sentiment_pos",
            "sentiment_neg", "sentiment_neu", "sentiment_compound"]
    cols = [c for c in cols if c in df.columns]
    out_df = df[cols]
    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Sentiment features computed:", len(out_df))


if __name__ == "__main__":
    main()
