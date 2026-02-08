# ==========================================
# PRODUCTION-READY FLASK BACKEND
# Groundwater Level Prediction API
# ==========================================

import os
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
from flask import render_template

# -------------------------------
# 1. Logging Setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------
# 2. Flask App Setup
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# 3. Required File Paths
# -------------------------------
MODEL_PATH = "models/ml_model.keras"
DATA_SCALER_PATH = "models/data_scaler.pkl"
Y_SCALER_PATH = "models/y_scaler.pkl"
DATASET_PATH = "data/groundwater_dataset.csv"

# -------------------------------
# 4. Load Model & Scalers
# -------------------------------
model = None
data_scaler = None
y_scaler = None

try:
    if not all(os.path.exists(p) for p in [MODEL_PATH, DATA_SCALER_PATH, Y_SCALER_PATH]):
        raise FileNotFoundError("Model or scaler files missing!")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    data_scaler = joblib.load(DATA_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    logging.info("Model and scalers loaded successfully")

except Exception as e:
    logging.error(f"Startup error: {e}")

# -------------------------------
# 5. Clean API → Dataset Mapping
# -------------------------------
FEATURE_MAP = {
    "rainfall_mm": "Rainfall (mm)",
    "soil_moisture": "Soil Moisture (%)",
    "evaporation_rate": "Evaporation Rate (mm/day)",
    "recharge_rate": "Recharge Rate (mm/year)",
    "well_yield": "Well Yield (L/s)",
    "aquifer_thickness": "Aquifer Thickness (m)"
}

# -------------------------------
# 6. Load Dataset Ranges
# -------------------------------
df = pd.read_csv(DATASET_PATH)

FEATURE_RANGES = {
    col: (df[col].min(), df[col].max())
    for col in FEATURE_MAP.values()
}

# -------------------------------
# 7. Dynamic Thresholds (from dataset)
# -------------------------------
LOW_THRESHOLD = df["Groundwater Level (m)"].quantile(0.33)
HIGH_THRESHOLD = df["Groundwater Level (m)"].quantile(0.66)

logging.info(f"Thresholds → Low < {LOW_THRESHOLD:.2f}, High > {HIGH_THRESHOLD:.2f}")

# -------------------------------
# 8. Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not available"}), 500

    try:
        data = request.get_json()
        input_dict = {}

        # -----------------------
        # Validate Inputs
        # -----------------------
        for api_key, dataset_col in FEATURE_MAP.items():

            if api_key not in data:
                return jsonify({"error": f"Missing parameter: {api_key}"}), 400

            try:
                value = float(data[api_key])
            except:
                return jsonify({"error": f"Invalid value for {api_key}"}), 400

            # Range validation (dataset-based)
            min_val, max_val = FEATURE_RANGES[dataset_col]
            if not (min_val <= value <= max_val):
                return jsonify({
                    "error": f"{dataset_col} out of realistic range "
                             f"({min_val:.2f} – {max_val:.2f})"
                }), 400

            input_dict[dataset_col] = value

        # -----------------------
        # Convert → DataFrame
        # -----------------------
        input_df = pd.DataFrame([input_dict])

        # -----------------------
        # Scale + Predict
        # -----------------------
        input_scaled = data_scaler.transform(input_df)
        pred_scaled = model.predict(input_scaled)
        groundwater_level = float(
            y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        )

        # -----------------------
        # Status Classification
        # -----------------------
        if groundwater_level < LOW_THRESHOLD:
            status = "Low"
        elif groundwater_level > HIGH_THRESHOLD:
            status = "High"
        else:
            status = "Normal"

        logging.info(f"Prediction: {groundwater_level:.2f} m")

        return jsonify({
            "groundwater_level": round(groundwater_level, 2),
            "status": status,
            "thresholds": {
                "low_below": round(LOW_THRESHOLD, 2),
                "high_above": round(HIGH_THRESHOLD, 2)
            }
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500


# -------------------------------
# 9. Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
