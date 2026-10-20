from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_churn_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_data.csv"
)


def main():
    print("Loading trained Logistic Regression model...")

    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_names = [
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Quantity",
        "Unique_Products",
        "Observation_Days",
    ]

    # Retrieve the trained Logistic Regression estimator.
    model = pipeline.named_steps["model"]

    # Coefficients correspond to standardized feature values.
    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
    })

    coefficients["Absolute_Importance"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute_Importance",
        ascending=False,
    )

    print("\n--- Feature Coefficients ---")
    print(
        coefficients[
            ["Feature", "Coefficient"]
        ].round(4).to_string(index=False)
    )

    print("\n--- Interpretation ---")

    for _, row in coefficients.iterrows():
        direction = (
            "higher predicted churn odds"
            if row["Coefficient"] > 0
            else "lower predicted churn odds"
        )

        print(
            f"{row['Feature']}: {direction} "
            f"(coefficient = {row['Coefficient']:.4f})"
        )

    print(
        "\nNote: coefficients describe associations in this "
        "dataset. They do not establish causation, and the "
        "results depend on the chosen churn definition."
    )

    print("\nModel analysis complete.")


if __name__ == "__main__":
    main()