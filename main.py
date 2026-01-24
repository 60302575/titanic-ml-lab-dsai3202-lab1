import argparse
from src.train import train_model
from src.test import predict_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic ML Project - Main entry point")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="'train' to fit and save pipeline; 'test' to generate predictions")
    args = parser.parse_args()

    if args.mode == "train":
        print("[MAIN] Training mode selected.")
        train_model()
    else:
        print("[MAIN] Test mode selected.")
        predict_test()
