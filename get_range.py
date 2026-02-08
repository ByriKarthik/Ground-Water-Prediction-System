import pandas as pd

df = pd.read_csv("groundwater_dataset.csv")

columns = [
    "Rainfall (mm)",
    "Soil Moisture (%)",
    "Evaporation Rate (mm/day)",
    "Recharge Rate (mm/year)",
    "Well Yield (L/s)",
    "Aquifer Thickness (m)"
]

print("\n=== DATASET RANGES ===\n")

for col in columns:
    print(f"{col}: {df[col].min():.2f}  →  {df[col].max():.2f}")
