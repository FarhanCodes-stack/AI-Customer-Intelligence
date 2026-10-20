from pathlib import Path

import numpy as np
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


def main():
    print("Loading customer RFM data...")

    df = pd.read_csv(RFM_PATH)

    features = ["Recency", "Frequency", "Monetary"]
    X = df[features].copy()

    # Apply the same transformations used previously.
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in [3, 4]:
        print(f"\n{'=' * 50}")
        print(f"RESULTS FOR {k} CLUSTERS")
        print("=" * 50)

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        df["Cluster"] = model.fit_predict(X_scaled)

        summary = df.groupby("Cluster").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        ).round(2)

        summary["Customer_Percentage"] = (
            summary["Customers"] / len(df) * 100
        ).round(2)

        print(summary.to_string())

    print("\nComparison complete.")


if __name__ == "__main__":
    main()