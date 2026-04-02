import argparse
import os
import time
import json

import azureml.mlflow
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)

SBERT_PREFIX   = "sbert_"
TFIDF_PREFIX   = "tfidf_"
SENTIMENT_COLS = ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]
LENGTH_COLS    = ["review_length_words", "review_length_chars"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data",   type=str, required=True)
    parser.add_argument("--test_data",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--C",        type=float, default=1.0)
    parser.add_argument("--max_iter", type=int,   default=1000)
    parser.add_argument("--solver",   type=str,   default="saga")
    return parser.parse_args()


def load_data(path):
    p = path if path.endswith(".parquet") else os.path.join(path, "data.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Parquet not found: {p}")
    return pd.read_parquet(p)


def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' missing. You had one job.")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


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

    X = np.hstack(parts)
    if X.shape[0] == 0:
        raise RuntimeError("Feature matrix is empty.")
    return X


def evaluate(model, X, y, split):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, proba)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    mlflow.log_metric(f"{split}_accuracy",  acc)
    mlflow.log_metric(f"{split}_auc",       auc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall",    rec)
    mlflow.log_metric(f"{split}_f1",        f1)
    print(f"[{split}] acc={acc:.4f} auc={auc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")


def main():
    args = parse_args()
    start_time = time.time()
    mlflow.start_run()

    mlflow.log_param("C",        args.C)
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("solver",   args.solver)

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df   = load_data(args.val_data)
    test_df  = load_data(args.test_data)
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"  columns: {train_df.columns.tolist()}")

    train_df = create_labels(train_df)
    val_df   = create_labels(val_df)
    test_df  = create_labels(test_df)

    print("Building features...")
    X_train = build_features(train_df); y_train = train_df["label"]
    X_val   = build_features(val_df);   y_val   = val_df["label"]
    X_test  = build_features(test_df);  y_test  = test_df["label"]
    print(f"  Feature matrix: {X_train.shape}")

    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train",    X_train.shape[0])

    print("Training Logistic Regression...")
    model = LogisticRegression(C=args.C, max_iter=args.max_iter,
                               solver=args.solver, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val,   y_val,   "val")
    evaluate(model, X_test,  y_test,  "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    sbert_cols = sorted([c for c in train_df.columns if c.startswith(SBERT_PREFIX)],
                        key=lambda x: int(x.split("_")[1]))
    meta = {
        "sbert_cols":     sbert_cols,
        "tfidf_cols":     [c for c in train_df.columns if c.startswith(TFIDF_PREFIX)],
        "sentiment_cols": [c for c in SENTIMENT_COLS if c in train_df.columns],
        "length_cols":    [c for c in LENGTH_COLS    if c in train_df.columns],
    }
    meta_path = os.path.join(args.output, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    mlflow.log_artifact(meta_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"Done. Runtime: {runtime:.1f}s")
    mlflow.end_run()


if __name__ == "__main__":
    main()
