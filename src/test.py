import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datetime import datetime


def predict_test(data_dir: str = "data", models_dir: str = "models", outputs_dir: str = "outputs") -> None:
    model_path = os.path.join(models_dir, "titanic_pipeline.joblib")
    test_csv = os.path.join(data_dir, "test.csv")
    targets_csv = os.path.join(data_dir, "gender_submission.csv")

    print(f"[TEST] Loading pipeline: {model_path}")
    pipe = joblib.load(model_path)
    print("[TEST] Pipeline loaded.")

    # Load test data
    df = pd.read_csv(test_csv)
    passenger_ids = df["PassengerId"].copy()
    X_test = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    # Load actual targets
    print(f"[TEST] Loading actual targets: {targets_csv}")
    targets_df = pd.read_csv(targets_csv)
    y_true = targets_df["Survived"].values

    print("[TEST] Predicting...")
    preds = pipe.predict(X_test)

    # Create output DataFrame with predictions and actual targets
    out = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds,
        "Actual": y_true
    })

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)

    # Print performance metrics
    print("\n" + "="*50)
    print("TEST PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, preds))
    print("="*50)

    # Save predictions
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"[TEST] Wrote predictions to {out_path}")

    # Save performance metrics to text file
    metrics_path = os.path.join(outputs_dir, "test_performance.txt")
    with open(metrics_path, 'w') as f:
        f.write("TITANIC ML MODEL - TEST PERFORMANCE METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, preds))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, preds)))
        f.write("\n\nDetailed Metrics:\n")
        f.write(f"True Positives:  {confusion_matrix(y_true, preds)[1, 1]}\n")
        f.write(f"True Negatives:  {confusion_matrix(y_true, preds)[0, 0]}\n")
        f.write(f"False Positives: {confusion_matrix(y_true, preds)[0, 1]}\n")
        f.write(f"False Negatives: {confusion_matrix(y_true, preds)[1, 0]}\n")

    print(f"[TEST] Saved performance metrics to {metrics_path}")
