import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()
    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val)
    test_df = pd.read_parquet(args.test)

    vectorizer = TfidfVectorizer(max_features=args.max_features, stop_words="english", ngram_range=(1, 1))
    train_tfidf = vectorizer.fit_transform(train_df["reviewText"].fillna(""))
    val_tfidf = vectorizer.transform(val_df["reviewText"].fillna(""))
    test_tfidf = vectorizer.transform(test_df["reviewText"].fillna(""))

    feature_names = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]

    for split_df, matrix, out_path in [
        (train_df, train_tfidf, args.train_out),
        (val_df, val_tfidf, args.val_out),
        (test_df, test_tfidf, args.test_out),
    ]:
        # Convert to dense numpy array then to dataframe
        out_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
        out_df["asin"] = split_df["asin"].values
        out_df["reviewerID"] = split_df["reviewerID"].values
        os.makedirs(out_path, exist_ok=True)
        out_df.to_parquet(os.path.join(out_path, "data.parquet"))
        del out_df

    print("TF-IDF done. Features:", len(feature_names))

if __name__ == "__main__":
    main()
