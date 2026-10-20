from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

    # Select the features used for customer segmentation.
    features = ["Recency", "Frequency", "Monetary"]
    X = df[features].copy()

    # Apply the same transformations as the original model.
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    # Standardize the features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nEvaluating different cluster counts...")
    print("This may take a little time.\n")

    results = []

    # Test between 2 and 6 clusters.
    for k in range(2, 7):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(2000, len(X_scaled)),
            random_state=42,
        )

        results.append({
            "Clusters": k,
            "Silhouette Score": score,
            "Inertia": model.inertia_,
        })

        print(f"Clusters: {k}")
        print(f"Silhouette Score: {score:.4f}")
        print(f"Inertia: {model.inertia_:.2f}\n")

    results_df = pd.DataFrame(results)

    best_result = results_df.loc[
        results_df["Silhouette Score"].idxmax()
    ]

    print("--- Evaluation Summary ---")
    print(results_df.round(4).to_string(index=False))

    print(
        "\nHighest silhouette score:",
        int(best_result["Clusters"]),
        "clusters",
    )

    print(
        "Score:",
        round(best_result["Silhouette Score"], 4),
    )

    print(
        "\nRemember: the highest silhouette score is a useful "
        "signal, not the only factor in choosing the final model."
    )


if __name__ == "__main__":
    main()