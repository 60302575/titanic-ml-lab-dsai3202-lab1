import json
import os
import numpy as np
import joblib

model        = None
feature_meta = None


def init():
    global model, feature_meta
    model_dir  = os.environ.get("AZUREML_MODEL_DIR", ".")
    
    # Find model.pkl
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        # Search subdirectories
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f == "model.pkl":
                    model_path = os.path.join(root, f)
                    break
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Find feature_meta.json
    meta_path = os.path.join(model_dir, "feature_meta.json")
    if not os.path.exists(meta_path):
        for root, dirs, files in os.walk(model_dir):
            for f in files:
                if f == "feature_meta.json":
                    meta_path = os.path.join(root, f)
                    break

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            feature_meta = json.load(f)
        print(f"Feature meta loaded: {list(feature_meta.keys())}")
    else:
        print("No feature_meta.json found")

    print("Model loaded successfully.")


def run(raw_data):
    try:
        data  = json.loads(raw_data)
        
        if "features" in data:
            X = np.array(data["features"], dtype=np.float32)
        else:
            return {"error": "Please send data as {'features': [[...], ...]}"}

        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return {
            "predictions":   preds.tolist(),
            "probabilities": proba.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
