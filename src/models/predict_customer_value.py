from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_purchase_classifier.joblib"
)

REVENUE_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_positive_revenue_model.joblib"
)

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]


def predict_customer_value(customer_features):
    """
    Predict purchase probability, conditional revenue,
    and expected future revenue for one customer.
    """

    classifier = joblib.load(CLASSIFIER_PATH)
    revenue_model = joblib.load(REVENUE_MODEL_PATH)

    customer_df = pd.DataFrame(
        [customer_features],
        columns=FEATURES,
    )

    purchase_probability = classifier.predict_proba(
        customer_df
    )[0, 1]

    conditional_revenue = np.expm1(
        revenue_model.predict(customer_df)[0]
    )

    conditional_revenue = max(
        float(conditional_revenue),
        0.0,
    )

    expected_revenue = (
        purchase_probability * conditional_revenue
    )

    return {
        "purchase_probability": float(purchase_probability),
        "conditional_revenue": conditional_revenue,
        "expected_revenue": float(expected_revenue),
    }


def main():
    print("Loading customer data and trained models...")

    df = pd.read_csv(DATA_PATH)

    # Choose an actual customer from the prepared dataset.
    customer = df.iloc[0]

    customer_features = {
        feature: float(customer[feature])
        for feature in FEATURES
    }

    prediction = predict_customer_value(customer_features)

    print("\n--- Customer Historical Features ---")

    for feature, value in customer_features.items():
        print(f"{feature}: {value:,.2f}")

    print("\n--- Predicted Customer Value ---")

    print(
        "Purchase probability: "
        f"{prediction['purchase_probability']:.1%}"
    )

    print(
        "Estimated revenue if the customer purchases: "
        f"£{prediction['conditional_revenue']:,.2f}"
    )

    print(
        "Expected future revenue: "
        f"£{prediction['expected_revenue']:,.2f}"
    )

    print("\nPrediction complete.")


if __name__ == "__main__":
    main()