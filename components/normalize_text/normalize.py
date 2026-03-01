import argparse
import os
import re
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    df["reviewText"] = df["reviewText"].fillna("").apply(normalize_text)
    df = df[df["reviewText"].str.len() >= 10]
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Normalized rows:", len(df))

if __name__ == "__main__":
    main()
