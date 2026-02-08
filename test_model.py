# ==========================================
# FINAL INFERENCE FILE (PRODUCTION READY)
# Groundwater Level Prediction
# ==========================================

import os
import numpy as np
import joblib
from tensorflow import keras
import pandas as pd

# -------------------------------
# 1. Check Required Files
# -------------------------------
MODEL_PATH = "ml_model.keras"
DATA_SCALER_PATH = "data_scaler.pkl"
Y_SCALER_PATH = "y_scaler.pkl"

for path in [MODEL_PATH, DATA_SCALER_PATH, Y_SCALER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

# -------------------------------
# 2. Load Model & Scalers
# -------------------------------
print("Loading model and scalers...")

model = keras.models.load_model(MODEL_PATH, compile=False)
data_scaler = joblib.load(DATA_SCALER_PATH)
y_scaler = joblib.load(Y_SCALER_PATH)

print("Model & scalers loaded successfully")

# -------------------------------
# 3. Define Feature Order (IMPORTANT)
# Must match training exactly
# -------------------------------
FEATURE_ORDER = [
    "Rainfall (mm)",
    "Soil Moisture (%)",
    "Evaporation Rate (mm/day)",
    "Recharge Rate (mm/year)",
    "Well Yield (L/s)",
    "Aquifer Thickness (m)"
]

EXPECTED_FEATURES = len(FEATURE_ORDER)

# -------------------------------
# 4. Sample Input (6 FEATURES ONLY)
# Replace with your own values
# -------------------------------
sample_input = pd.DataFrame([{
    "Rainfall (mm)": 1677.76,
    "Soil Moisture (%)": 30.61,
    "Evaporation Rate (mm/day)": 6.96,
    "Recharge Rate (mm/year)": 185.96,
    "Well Yield (L/s)": 22.26,
    "Aquifer Thickness (m)": 50.35
}])

# -------------------------------
# 5. Validate Feature Count
# -------------------------------
if sample_input.shape[1] != EXPECTED_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_FEATURES} features, but got {sample_input.shape[1]}"
    )

# -------------------------------
# 6. Scale Input
# -------------------------------
input_scaled = data_scaler.transform(sample_input)

# -------------------------------
# 7. Predict
# -------------------------------
pred_scaled = model.predict(input_scaled)
predicted_level = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

# -------------------------------
# 8. Output
# -------------------------------
print("\n===== PREDICTION RESULT =====")
print(f"Predicted Groundwater Level: {predicted_level:.2f} meters")

# -------------------------------
# OPTIONAL: Batch Prediction Example
# -------------------------------
"""
batch_input = np.array([
    [1600, 28, 7, 180, 20, 48],
    [1700, 32, 6, 200, 23, 52]
])

batch_scaled = data_scaler.transform(batch_input)
batch_pred = model.predict(batch_scaled)
batch_pred = y_scaler.inverse_transform(batch_pred)

print("\nBatch Predictions:")
for i, val in enumerate(batch_pred):
    print(f"Sample {i+1}: {val[0]:.2f} meters")
"""
