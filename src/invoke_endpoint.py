import argparse
import json
import os
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SBERT_PREFIX   = "sbert_"
TFIDF_PREFIX   = "tfidf_"
SENTIMENT_COLS = ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]
LENGTH_COLS    = ["review_length_words", "review_length_chars"]


def build_features(df):
    parts = []
    sbert_cols = sorted([c for c in df.columns if c.startswith(SBERT_PREFIX)],
                        key=lambda x: int(x.split("_")[1]))
    if not sbert_cols:
        raise RuntimeError(f"No sbert_* columns found. Columns: {df.columns.tolist()}")
    parts.append(df[sbert_cols].values.astype(np.float32))
    tfidf_cols = [c for c in df.columns if c.startswith(TFIDF_PREFIX)]
    if tfidf_cols:
        parts.append(df[tfidf_cols].values.astype(np.float32))
    sent_cols = [c for c in SENTIMENT_COLS if c in df.columns]
    if sent_cols:
        parts.append(df[sent_cols].values.astype(np.float32))
    len_cols = [c for c in LENGTH_COLS if c in df.columns]
    if len_cols:
        parts.append(df[len_cols].values.astype(np.float32))
    return np.hstack(parts)


def invoke_in_batches(X, endpoint_url, api_key, batch_size=100):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    all_preds = []
    n_batches = (len(X) + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch  = X[i * batch_size:(i + 1) * batch_size]
        resp   = requests.post(endpoint_url, headers=headers,
                               data=json.dumps({"features": batch.tolist()}), timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            raise RuntimeError(f"Endpoint error: {result['error']}")
        all_preds.extend(result["predictions"])
        print(f"  Batch {i+1}/{n_batches} done ({len(batch)} rows)")
    return np.array(all_preds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_data",  type=str, required=True)
    parser.add_argument("--endpoint_url", type=str, default=os.environ.get("ENDPOINT_URL", ""))
    parser.add_argument("--api_key",      type=str, default=os.environ.get("API_KEY", ""))
    parser.add_argument("--batch_size",   type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.deploy_data
    if os.path.isdir(path):
        path = os.path.join(path, "data.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Deploy data not found: {path}")

    print(f"Loading: {path}")
    df     = pd.read_parquet(path)
    X      = build_features(df)
    y_true = (df["overall"] >= 4).astype(int).values
    print(f"  Feature matrix: {X.shape}")

    print("Invoking endpoint...")
    preds = invoke_in_batches(X, args.endpoint_url, args.api_key, args.batch_size)

    print("\n=== Deployment Evaluation ===")
    print(f"  Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"  F1       : {f1_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_true, preds, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
