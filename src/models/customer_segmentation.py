import numpy as np
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RFM_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_rfm.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_segments.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kmeans_model.joblib"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rfm_scaler.joblib"
)


def main():
    print("Loading customer RFM data...")

    df = pd.read_csv(RFM_PATH)

    features = ["Recency", "Frequency", "Monetary"]
    X = df[features].copy()

    # Reduce the influence of extreme values.
    X["Frequency"] = X["Frequency"].clip(lower=1)
    X["Monetary"] = X["Monetary"].clip(lower=0)

    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    # Standardize the features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training K-Means model...")

    # Start with four customer clusters.
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    # Save customer assignments.
    df.to_csv(OUTPUT_PATH, index=False)

    # Save the trained model and scaler.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\n--- Customers Per Cluster ---")
    print(df["Cluster"].value_counts().sort_index())

    print("\n--- Cluster Profiles ---")
    print(
        df.groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    print("\n--- Saved Files ---")
    print(f"Customer segments: {OUTPUT_PATH}")
    print(f"K-Means model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")


if __name__ == "__main__":
    main()