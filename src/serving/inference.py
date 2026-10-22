"""Inference from an explicitly exported XGBoost model; no MLflow server needed."""
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "bundle")))
MANIFEST = json.loads((MODEL_DIR / "manifest.json").read_text())
if MANIFEST["schema_version"] != 1:
    raise ValueError("Unsupported serving manifest schema")
if hashlib.sha256((MODEL_DIR / "model.ubj").read_bytes()).hexdigest() != MANIFEST["model_sha256"]:
    raise ValueError("Model checksum does not match manifest")
FEATURE_COLS = MANIFEST["feature_columns"]
THRESHOLD = float(MANIFEST["threshold"])
if not 0 < THRESHOLD < 1:
    raise ValueError("Invalid model threshold")
model = XGBClassifier(n_jobs=1)
model.load_model(MODEL_DIR / "model.ubj")
model.set_params(n_jobs=1)
if model.get_booster().feature_names != FEATURE_COLS:
    raise ValueError("Model feature order does not match manifest")

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    **{c: {"No": 0, "Yes": 1} for c in
       ("Partner", "Dependents", "PhoneService", "PaperlessBilling")},
}
CATEGORIES = {
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: ["No", "No internet service", "Yes"] for c in
       ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
}
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Use the full training category vocabulary even for a single customer."""
    df = df.copy()
    required = list(BINARY_MAP) + list(CATEGORIES) + NUMERIC_COLS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing customer fields: {sorted(missing)}")
    for column, mapping in BINARY_MAP.items():
        if not df[column].isin(mapping).all():
            raise ValueError(f"Invalid category for {column}")
        df[column] = df[column].map(mapping).astype(int)
    for column, categories in CATEGORIES.items():
        if not df[column].isin(categories).all():
            raise ValueError(f"Invalid category for {column}")
        df[column] = pd.Categorical(df[column], categories=categories)
    for column in NUMERIC_COLS:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if df[column].isna().any():
            raise ValueError(f"Missing numeric value for {column}")
    df = pd.get_dummies(df, columns=list(CATEGORIES), drop_first=True, dtype=int)
    return df.reindex(columns=FEATURE_COLS, fill_value=0)


def predict_details(input_dict: dict) -> dict:
    probability = float(model.predict_proba(_serve_transform(pd.DataFrame([input_dict])))[0, 1])
    return {
        "prediction": "Likely to churn" if probability >= THRESHOLD else "Not likely to churn",
        "churn_probability": probability,
        "threshold": THRESHOLD,
    }


def predict(input_dict: dict) -> str:
    return predict_details(input_dict)["prediction"]
