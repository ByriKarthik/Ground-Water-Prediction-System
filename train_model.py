# ===============================
# FINAL PRODUCTION TRAINING FILE
# Groundwater Level Prediction
# ===============================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# -------------------------------
# 1. Reproducibility (Deterministic)
# -------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------
# 2. Load Dataset (NO hardcoded path)
# -------------------------------
DATASET_PATH = "groundwater_dataset.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset not found! Place 'groundwater_dataset.csv' in project folder.")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------
# 3. Remove Outliers (IQR Method)
# -------------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f"After outlier removal: {df.shape[0]} rows")

# -------------------------------
# 4. Define FINAL 6 Features (NO Permeability)
# -------------------------------
target_col = "Groundwater Level (m)"

selected_features = [
    "Rainfall (mm)",
    "Soil Moisture (%)",
    "Evaporation Rate (mm/day)",
    "Recharge Rate (mm/year)",
    "Well Yield (L/s)",
    "Aquifer Thickness (m)"
]

for col in selected_features + [target_col]:
    if col not in df.columns:
        raise ValueError(f"Column missing in dataset: {col}")

X = df[selected_features]
y = df[target_col].values.reshape(-1, 1)

print("Final Features Used:", selected_features)

# -------------------------------
# 5. Scaling
# -------------------------------
data_scaler = MinMaxScaler()
X_scaled = data_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y).flatten()

joblib.dump(data_scaler, "data_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")
print("Scalers saved")

# -------------------------------
# 6. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# -------------------------------
# 7. Model Definition
# -------------------------------
def create_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=l1(0.005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu', kernel_regularizer=l1(0.005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

model = create_model()

# -------------------------------
# 8. Training Callbacks
# -------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=6,
    min_lr=1e-6
)

# -------------------------------
# 9. Train Model
# -------------------------------
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=120,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

# -------------------------------
# 10. Save Model
# -------------------------------
model.save("ml_model.keras")
print("Model saved: ml_model.keras")

# -------------------------------
# 11. Evaluate Model
# -------------------------------
y_pred_scaled = model.predict(X_test).reshape(-1, 1)

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# -------------------------------
# 12. Save Training Graph
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_loss.png")
print("Training graph saved: training_loss.png")
