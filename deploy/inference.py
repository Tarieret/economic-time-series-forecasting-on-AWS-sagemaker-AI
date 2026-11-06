import os, json
import joblib
import pandas as pd
import xgboost as xgb

def model_fn(model_dir):
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "xgboost_model.json"))
    features = joblib.load(os.path.join(model_dir, "features.pkl"))
    return {"model": booster, "features": features}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    # input: {"last_value": float, "recent_changes": [12 monthly changes, oldest first]}
    row = pd.DataFrame([input_data["recent_changes"][::-1]], columns=model_dict["features"])
    d = float(model_dict["model"].predict(xgb.DMatrix(row))[0])
    return {"predicted_change": d, "predicted_level": input_data["last_value"] + d}

def output_fn(prediction, accept):
    return json.dumps(prediction)
