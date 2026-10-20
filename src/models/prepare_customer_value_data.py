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
    / "customer_value_data.csv"
)

OBSERVATION_START = pd.Timestamp("2010-12-01")
OBSERVATION_END = pd.Timestamp("2011-06-30 23:59:59")

PREDICTION_START = pd.Timestamp("2011-07-01")
PREDICTION_END = pd.Timestamp("2011-09-30 23:59:59")


def main():
    print("Loading cleaned transaction data...")

    df = pd.read_csv(DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Historical observation period.
    observation = df[
        (df["InvoiceDate"] >= OBSERVATION_START)
        & (df["InvoiceDate"] <= OBSERVATION_END)
    ].copy()

    # Future prediction period.
    future = df[
        (df["InvoiceDate"] >= PREDICTION_START)
        & (df["InvoiceDate"] <= PREDICTION_END)
    ].copy()

    print("\n--- Window Summary ---")
    print(f"Observation rows: {len(observation):,}")
    print(f"Prediction rows: {len(future):,}")

    # Build all predictive features using historical data only.
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
    )

    # Build the target from the future period only.
    future_revenue = (
        future.groupby("CustomerID")["Revenue"]
        .sum()
        .rename("Future_Revenue")
    )

    features = features.join(future_revenue, how="left")

    features["Future_Revenue"] = (
        features["Future_Revenue"].fillna(0)
    )

    features = features.reset_index()

    # Basic data integrity checks.
    assert features["CustomerID"].is_unique
    assert features["Recency"].ge(0).all()
    assert features["Frequency"].ge(1).all()
    assert features["Monetary"].gt(0).all()
    assert features["Total_Quantity"].gt(0).all()
    assert features["Unique_Products"].ge(1).all()
    assert features["Future_Revenue"].ge(0).all()

    features.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Dataset Summary ---")
    print(f"Eligible customers: {len(features):,}")

    print(
        "Customers with future purchases: "
        f"{int((features['Future_Revenue'] > 0).sum()):,}"
    )

    print(
        "Customers with zero future revenue: "
        f"{int((features['Future_Revenue'] == 0).sum()):,}"
    )

    print(
        f"Total future revenue: "
        f"£{features['Future_Revenue'].sum():,.2f}"
    )

    print(
        f"Average future revenue per eligible customer: "
        f"£{features['Future_Revenue'].mean():,.2f}"
    )

    print("\n--- Observation Window Verification ---")
    print(
        "Latest observation transaction: "
        f"{observation['InvoiceDate'].max()}"
    )

    print(
        "Earliest prediction transaction: "
        f"{future['InvoiceDate'].min()}"
    )

    print(
        "Latest prediction transaction: "
        f"{future['InvoiceDate'].max()}"
    )

    print("\n--- Feature and Target Validation ---")
    print("Customer IDs are unique: Yes")
    print("Historical features are nonnegative: Yes")
    print("Future revenue contains no negative values: Yes")
    print("All validation checks passed.")

    print(f"\nSaved dataset to: {OUTPUT_PATH}")
    print("\nCustomer value dataset preparation complete.")


if __name__ == "__main__":
    main()