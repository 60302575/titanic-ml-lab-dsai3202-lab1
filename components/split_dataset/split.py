import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str,   required=True)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--test_ratio",  type=float, default=0.15)
    parser.add_argument("--train_out",   type=str,   required=True)
    parser.add_argument("--val_out",     type=str,   required=True)
    parser.add_argument("--test_out",    type=str,   required=True)
    parser.add_argument("--deploy_out",  type=str,   required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_parquet(args.data)

    if "review_year" not in df.columns and "unixReviewTime" in df.columns:
        df["review_year"] = pd.to_datetime(df["unixReviewTime"], unit="s").dt.year
    elif "review_year" not in df.columns and "reviewTime" in df.columns:
        df["review_year"] = pd.to_datetime(df["reviewTime"]).dt.year

    if "review_year" in df.columns:
        df_sorted    = df.sort_values("review_year", ascending=True).reset_index(drop=True)
        deploy_size  = int(len(df_sorted) * 0.10)
        deploy_df    = df_sorted.iloc[-deploy_size:]
        remaining_df = df_sorted.iloc[:-deploy_size]
    else:
        print("WARNING: review_year not found, using random deploy split")
        remaining_df, deploy_df = train_test_split(
            df, test_size=0.10, random_state=args.seed, shuffle=True
        )

    total      = args.train_ratio + args.val_ratio + args.test_ratio
    train_frac = args.train_ratio / total
    val_frac   = args.val_ratio   / total

    train_df, temp_df = train_test_split(
        remaining_df, test_size=(1 - train_frac), random_state=args.seed, shuffle=True
    )
    val_size = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size), random_state=args.seed, shuffle=True
    )

    for out_path, split_df, name in [
        (args.train_out,  train_df,  "train"),
        (args.val_out,    val_df,    "val"),
        (args.test_out,   test_df,   "test"),
        (args.deploy_out, deploy_df, "deploy"),
    ]:
        os.makedirs(out_path, exist_ok=True)
        split_df.to_parquet(os.path.join(out_path, "data.parquet"), index=False)
        print(f"{name}: {len(split_df)} rows")


if __name__ == "__main__":
    main()
