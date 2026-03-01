import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    model = SentenceTransformer(args.model_name)
    texts = df["reviewText"].fillna("").tolist()
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"sbert_{i}" for i in range(embeddings.shape[1])])
    emb_df["asin"] = df["asin"].values
    emb_df["reviewerID"] = df["reviewerID"].values
    os.makedirs(args.out, exist_ok=True)
    emb_df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("SBERT embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    main()
