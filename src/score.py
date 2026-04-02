import json
import os
import numpy as np
import joblib

model        = None
feature_meta = None


def init():
    global model, feature_meta
    model_dir  = os.environ.get("AZUREML_MODEL_DIR", ".")
    model      = joblib.load(os.path.join(model_dir, "model.pkl"))
    meta_path  = os.path.join(model_dir, "feature_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            feature_meta = json.load(f)
    print("Model loaded.")


def _build_features(data):
    if "features" in data:
        return np.array(data["features"], dtype=np.float32)
    parts = []
    if feature_meta:
        for col in feature_meta.get("sbert_cols", []):
            parts.append(np.array(data[col], dtype=np.float32).reshape(-1, 1))
        for col in feature_meta.get("tfidf_cols", []):
            parts.append(np.array(data.get(col, [0.0]), dtype=np.float32).reshape(-1, 1))
        for col in feature_meta.get("sentiment_cols", []):
            parts.append(np.array(data.get(col, [0.0]), dtype=np.float32).reshape(-1, 1))
        for col in feature_meta.get("length_cols", []):
            parts.append(np.array(data.get(col, [0.0]), dtype=np.float32).reshape(-1, 1))
    if not parts:
        raise ValueError("Cannot build features. Send 'features' key with pre-built array.")
    return np.hstack(parts)


def run(raw_data):
    try:
        data  = json.loads(raw_data)
        X     = _build_features(data)
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return {"predictions": preds.tolist(), "probabilities": proba.tolist()}
    except Exception as e:
        return {"error": str(e)}
