from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_model.joblib"
)

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]

TARGET = "Future_Revenue"


def main():
    print("Loading customer value data and model...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = joblib.load(MODEL_PATH)

    predictions = np.maximum(
        model.predict(X_test),
        0,
    )

    results = pd.DataFrame({
        "Actual_Revenue": y_test.to_numpy(),
        "Predicted_Revenue": predictions,
    })

    print("\n--- Overall Revenue Comparison ---")

    print(
        f"Average actual revenue: "
        f"£{results['Actual_Revenue'].mean():,.2f}"
    )

    print(
        f"Average predicted revenue: "
        f"£{results['Predicted_Revenue'].mean():,.2f}"
    )

    print(
        f"Median actual revenue: "
        f"£{results['Actual_Revenue'].median():,.2f}"
    )

    print(
        f"Median predicted revenue: "
        f"£{results['Predicted_Revenue'].median():,.2f}"
    )

    print("\n--- Performance Across Actual Revenue Ranges ---")

    results["Revenue_Range"] = pd.cut(
        results["Actual_Revenue"],
        bins=[-1, 0, 100, 500, 2000, float("inf")],
        labels=[
            "£0",
            "£0.01–£100",
            "£100.01–£500",
            "£500.01–£2,000",
            "Above £2,000",
        ],
    )

    summary = (
        results.groupby("Revenue_Range", observed=False)
        .agg(
            Customers=("Actual_Revenue", "size"),
            Average_Actual=("Actual_Revenue", "mean"),
            Average_Predicted=("Predicted_Revenue", "mean"),
            MAE=(
                "Actual_Revenue",
                lambda actual: mean_absolute_error(
                    actual,
                    results.loc[actual.index, "Predicted_Revenue"],
                ),
            ),
        )
        .round(2)
    )

    print(summary.to_string())

    print("\n--- Highest Predicted Customer Values ---")

    print(
        results.nlargest(10, "Predicted_Revenue")
        .round(2)
        .to_string(index=False)
    )

    print("\nPrediction analysis complete.")


if __name__ == "__main__":
    main()