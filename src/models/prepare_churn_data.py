from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_retail_data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_data.csv"
)

OBSERVATION_START = pd.Timestamp("2010-12-01")
OBSERVATION_END = pd.Timestamp("2011-06-30 23:59:59")

PREDICTION_START = pd.Timestamp("2011-07-01")
PREDICTION_END = pd.Timestamp("2011-09-30 23:59:59")


def main():
    print("Loading cleaned transaction data...")

    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Purchases used to calculate customer features.
    observation = df[
        (df["InvoiceDate"] >= OBSERVATION_START)
        & (df["InvoiceDate"] <= OBSERVATION_END)
    ].copy()

    # Purchases used only to create the target label.
    prediction = df[
        (df["InvoiceDate"] >= PREDICTION_START)
        & (df["InvoiceDate"] <= PREDICTION_END)
    ].copy()

    print("\n--- Window Summary ---")
    print(f"Observation rows: {len(observation):,}")
    print(f"Prediction rows: {len(prediction):,}")

    # Every eligible customer must have purchased in the
    # observation window.
    features = observation.groupby("CustomerID").agg(
        Recency=(
            "InvoiceDate",
            lambda dates: (
                OBSERVATION_END.normalize()
                - dates.max().normalize()
            ).days,
        ),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Unique_Products=("StockCode", "nunique"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
    )

    # Customers who purchased during the prediction window.
    future_customers = set(prediction["CustomerID"].unique())

    # 1 = no purchase in the prediction window.
    # 0 = at least one purchase in the prediction window.
    features["Churn"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    features = features.reset_index()

    features["Observation_Days"] = (
        features["Last_Purchase"].dt.normalize()
        - features["First_Purchase"].dt.normalize()
    ).dt.days

    # Keep dates for auditing, but do not use them as model
    # features without a deliberate feature-engineering decision.
    features.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Churn Dataset Summary ---")
    print(f"Eligible customers: {len(features):,}")
    print(f"Customers labelled churned: {features['Churn'].sum():,}")
    print(
        "Customers who purchased in prediction window: "
        f"{(features['Churn'] == 0).sum():,}"
    )
    print(
        "Churn label rate: "
        f"{features['Churn'].mean() * 100:.2f}%"
    )

    print("\n--- Churn Label Counts ---")
    print(features["Churn"].value_counts().sort_index().to_string())

    print("\n--- Sample Rows ---")
    print(features.head().to_string(index=False))

    print(f"\nSaved churn dataset to: {OUTPUT_PATH}")
    print("\nChurn dataset preparation complete.")


if __name__ == "__main__":
    main()