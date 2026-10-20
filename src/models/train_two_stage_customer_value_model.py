from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]

TARGET = "Future_Revenue"


def main():
    print("Loading customer value data...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Stage 1: Predict whether the customer will purchase.
    print("\nTraining purchase classifier...")

    y_train_purchase = (y_train > 0).astype(int)

    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    classifier.fit(X_train, y_train_purchase)

    # Stage 2: Predict revenue for customers who purchase.
    print("Training positive-revenue regressor...")

    positive_mask = y_train > 0

    revenue_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    revenue_model.fit(
        X_train.loc[positive_mask],
        np.log1p(y_train.loc[positive_mask]),
    )

    # Evaluate the combined prediction.
    purchase_probability = classifier.predict_proba(X_test)[:, 1]

    conditional_revenue = np.expm1(
        revenue_model.predict(X_test)
    )

    predictions = np.maximum(
        purchase_probability * conditional_revenue,
        0,
    )

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Two-Stage Model Evaluation ---")
    print(f"Test customers: {len(y_test)}")
    print(f"MAE: £{mae:,.2f}")
    print(f"RMSE: £{rmse:,.2f}")
    print(f"Actual average revenue: £{y_test.mean():,.2f}")
    print(f"Predicted average revenue: £{predictions.mean():,.2f}")

    # Save the two components separately.
    classifier_path = (
        OUTPUT_DIR / "customer_purchase_classifier.joblib"
    )

    revenue_model_path = (
        OUTPUT_DIR / "customer_positive_revenue_model.joblib"
    )

    joblib.dump(classifier, classifier_path)
    joblib.dump(revenue_model, revenue_model_path)

    # Save a compact evaluation report.
    results = pd.DataFrame([
        {
            "Model": "Two-Stage Customer Value Model",
            "Test_Customers": len(y_test),
            "MAE": mae,
            "RMSE": rmse,
            "Actual_Average_Revenue": y_test.mean(),
            "Predicted_Average_Revenue": predictions.mean(),
        }
    ])

    results.to_csv(
        OUTPUT_DIR / "two_stage_customer_value_results.csv",
        index=False,
    )

    print("\nSaved classifier:", classifier_path)
    print("Saved revenue model:", revenue_model_path)
    print(
        "Saved evaluation:",
        OUTPUT_DIR / "two_stage_customer_value_results.csv",
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()