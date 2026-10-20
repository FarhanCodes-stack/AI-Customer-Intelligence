from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    print("Loading data and trained model...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    # Recreate the same test split used during training.
    X_train, X_test, y_train, y_test = train_test_split(
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

    results["Absolute_Error"] = (
        results["Actual_Revenue"]
        - results["Predicted_Revenue"]
    ).abs()

    print("\n--- Test Set Overview ---")
    print(f"Test customers: {len(results):,}")
    print(
        "Actual zero-revenue customers:",
        int((results["Actual_Revenue"] == 0).sum()),
    )
    print(
        "Actual paying customers:",
        int((results["Actual_Revenue"] > 0).sum()),
    )

    print("\n--- Overall Performance ---")
    print(
        f"MAE: £{mean_absolute_error(y_test, predictions):,.2f}"
    )
    print(
        f"RMSE: £{np.sqrt(mean_squared_error(y_test, predictions)):,.2f}"
    )

    print("\n--- Performance by Actual Future Revenue ---")

    results["Revenue_Group"] = np.where(
        results["Actual_Revenue"] == 0,
        "Zero revenue",
        "Positive revenue",
    )

    for group, group_df in results.groupby("Revenue_Group"):
        mae = group_df["Absolute_Error"].mean()

        print(
            f"{group}: {len(group_df)} customers, "
            f"MAE = £{mae:,.2f}"
        )

    print("\n--- Largest Prediction Errors ---")

    largest_errors = results.nlargest(
        10,
        "Absolute_Error",
    )

    print(
        largest_errors.round(2).to_string(index=False)
    )

    print("\n--- Prediction Distribution ---")
    print(
        results["Predicted_Revenue"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()