import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

DATA_PATH = Path("data/processed/customer_value_data.csv")
CLASSIFIER_PATH = Path(
    "data/processed/customer_purchase_classifier.joblib"
)
REGRESSOR_PATH = Path(
    "data/processed/customer_positive_revenue_model.joblib"
)
OUTPUT_PATH = Path(
    "data/processed/customer_value_prediction_report.csv"
)

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]


def main():
    print("Loading customer data and trained models...")

    data = pd.read_csv(DATA_PATH)

    classifier = load(CLASSIFIER_PATH)
    regressor = load(REGRESSOR_PATH)

    X = data[FEATURES]

    print("Generating predictions...")

    purchase_probability = classifier.predict_proba(X)[:, 1]

    conditional_revenue = np.expm1(
        regressor.predict(X)
    )

    expected_revenue = (
        purchase_probability * conditional_revenue
    )

    report = pd.DataFrame({
        "CustomerID": data["CustomerID"],
        "Recency": data["Recency"],
        "Frequency": data["Frequency"],
        "Historical_Revenue": data["Monetary"],
        "Purchase_Probability": purchase_probability,
        "Predicted_Revenue_If_Purchasing": conditional_revenue,
        "Expected_Future_Revenue": expected_revenue,
        "Actual_Future_Revenue": data["Future_Revenue"],
    })

    report["Prediction_Error"] = (
        report["Actual_Future_Revenue"]
        - report["Expected_Future_Revenue"]
    )

    report["Customer_Priority"] = pd.cut(
        report["Expected_Future_Revenue"],
        bins=[-np.inf, 100, 500, 2000, np.inf],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    report = report.sort_values(
        "Expected_Future_Revenue",
        ascending=False,
    )

    report.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Customer Value Prediction Report ---")
    print(f"Customers evaluated: {len(report)}")
    print(f"Report saved to: {OUTPUT_PATH}")

    print("\nTop 10 customers by expected future revenue:")

    print(
        report[
            [
                "CustomerID",
                "Purchase_Probability",
                "Expected_Future_Revenue",
                "Customer_Priority",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nPriority group counts:")

    print(
        report["Customer_Priority"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()