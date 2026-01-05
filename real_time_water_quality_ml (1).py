
# Real-Time Water Quality Tracking with ML-Integrated Sensors (Synthetic Data)
# ------------------------------------------------------------
# Author: ChatGPT
# Description:
# End-to-end example using synthetic data for real-time water quality monitoring
# Includes data generation, ML training, and live sensor simulation

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------- Synthetic Data Generator ----------------
def generate_synthetic_water_data(n_samples=10000, anomaly_ratio=0.05):
    np.random.seed(42)
    data = {
        "pH": np.random.normal(7.2, 0.3, n_samples),
        "turbidity": np.random.normal(3, 1, n_samples),
        "dissolved_oxygen": np.random.normal(8, 0.8, n_samples),
        "temperature": np.random.normal(20, 3, n_samples),
        "conductivity": np.random.normal(300, 50, n_samples),
        "nitrate": np.random.normal(2, 0.7, n_samples)
    }
    df = pd.DataFrame(data)

    anomaly_indices = np.random.choice(
        df.index, int(n_samples * anomaly_ratio), replace=False
    )

    df.loc[anomaly_indices, "pH"] += np.random.uniform(-2, 2)
    df.loc[anomaly_indices, "turbidity"] += np.random.uniform(10, 40)
    df.loc[anomaly_indices, "nitrate"] += np.random.uniform(10, 30)

    df["anomaly"] = 0
    df.loc[anomaly_indices, "anomaly"] = 1
    return df

# ---------------- Water Quality Index ----------------
def compute_wqi(row):
    weights = {
        "pH": 0.2,
        "turbidity": 0.2,
        "dissolved_oxygen": 0.25,
        "temperature": 0.1,
        "conductivity": 0.15,
        "nitrate": 0.1
    }
    scores = {
        "pH": max(0, 100 - abs(row["pH"] - 7) * 15),
        "turbidity": max(0, 100 - row["turbidity"] * 5),
        "dissolved_oxygen": min(100, row["dissolved_oxygen"] * 10),
        "temperature": max(0, 100 - abs(row["temperature"] - 20) * 3),
        "conductivity": max(0, 100 - row["conductivity"] / 5),
        "nitrate": max(0, 100 - row["nitrate"] * 4)
    }
    return sum(scores[p] * weights[p] for p in weights)

# ---------------- Train Models ----------------
print("Generating synthetic data...")
df = generate_synthetic_water_data()
df["WQI"] = df.apply(compute_wqi, axis=1)

features = ["pH", "turbidity", "dissolved_oxygen", "temperature", "conductivity", "nitrate"]
X = df[features]
y_wqi = df["WQI"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training anomaly detection model...")
anomaly_model = IsolationForest(contamination=0.05)
anomaly_model.fit(X_scaled)

print("Training WQI prediction model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_wqi, test_size=0.2)
wqi_model = RandomForestRegressor(n_estimators=100)
wqi_model.fit(X_train, y_train)

# ---------------- Real-Time Sensor Simulation ----------------
print("Starting real-time water quality monitoring...\nPress CTRL+C to stop.")

try:
    while True:
        sensor_reading = {
            "pH": np.random.normal(7.2, 0.4),
            "turbidity": np.random.normal(3, 1.2),
            "dissolved_oxygen": np.random.normal(8, 1),
            "temperature": np.random.normal(20, 4),
            "conductivity": np.random.normal(300, 60),
            "nitrate": np.random.normal(2, 0.8)
        }

        X_live = pd.DataFrame([sensor_reading])
        X_scaled_live = scaler.transform(X_live)

        anomaly = anomaly_model.predict(X_scaled_live)[0]
        wqi = wqi_model.predict(X_scaled_live)[0]

        status = "CONTAMINATION ALERT" if anomaly == -1 else "NORMAL"

        print("Sensor Data:", sensor_reading)
        print(f"WQI: {wqi:.2f} | Status: {status}")
        print("-" * 60)

        time.sleep(2)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")
