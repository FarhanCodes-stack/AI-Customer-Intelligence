from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)

EXISTING_MODEL_PATH = (
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


def evaluate_model(name, actual, predicted):
    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Actual_Average": actual.mean(),
        "Predicted_Average": predicted.mean(),
    }


def main():
    print("Loading customer value dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    results = []

    # Model 1: Mean baseline
    baseline_prediction = np.full(
        len(y_test),
        y_train.mean(),
    )

    results.append(
        evaluate_model(
            "Mean Baseline",
            y_test,
            baseline_prediction,
        )
    )

    # Model 2: Existing Random Forest
    existing_model = joblib.load(EXISTING_MODEL_PATH)

    existing_predictions = existing_model.predict(X_test)

    results.append(
        evaluate_model(
            "Existing Random Forest",
            y_test,
            existing_predictions,
        )
    )

    # Model 3: Two-stage model
    print("\nTraining purchase-probability classifier...")

    y_train_purchase = (y_train > 0).astype(int)
    y_test_purchase = (y_test > 0).astype(int)

    classifier = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    classifier.fit(X_train, y_train_purchase)

    purchase_probabilities = classifier.predict_proba(X_test)[:, 1]

    print(
        "Purchase classification accuracy:",
        round(
            accuracy_score(
                y_test_purchase,
                purchase_probabilities >= 0.5,
            ),
            4,
        ),
    )

    print(
        "Purchase classification ROC-AUC:",
        round(
            roc_auc_score(
                y_test_purchase,
                purchase_probabilities,
            ),
            4,
        ),
    )

    # Train the revenue regressor only on customers
    # who actually purchased during the future period.
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

    positive_revenue_predictions = np.expm1(
        revenue_model.predict(X_test)
    )

    # Combine purchase probability with predicted revenue
    # conditional on a purchase.
    two_stage_predictions = (
        purchase_probabilities
        * positive_revenue_predictions
    )

    results.append(
        evaluate_model(
            "Two-Stage Model",
            y_test,
            two_stage_predictions,
        )
    )

    comparison = pd.DataFrame(results)

    numeric_columns = [
        "MAE",
        "RMSE",
        "Actual_Average",
        "Predicted_Average",
    ]

    comparison[numeric_columns] = comparison[numeric_columns].round(2)

    print("\n--- Customer Value Model Comparison ---")
    print(comparison.to_string(index=False))

    best_model = comparison.loc[
        comparison["MAE"].idxmin(),
        "Model",
    ]

    print(f"\nBest model by MAE: {best_model}")
    print("\nComparison complete.")


if __name__ == "__main__":
    main()