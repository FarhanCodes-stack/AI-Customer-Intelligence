from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)


def main():
    print("Loading customer value dataset...")

    df = pd.read_csv(DATA_PATH)

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "Total_Quantity",
        "Unique_Products",
        "Future_Revenue",
    ]

    print("\n--- Dataset Overview ---")
    print(f"Rows: {len(df):,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")

    print("\n--- Feature Summary ---")
    print(df[features].describe().round(2).to_string())

    print("\n--- Customers With Highest Historical Monetary Values ---")

    top_customers = df.nlargest(10, "Monetary")

    print(
        top_customers[
            ["CustomerID"] + features
        ].round(2).to_string(index=False)
    )

    print("\n--- Customers With One Purchase and High Monetary Values ---")

    unusual_customers = df[
        (df["Frequency"] == 1)
        & (df["Monetary"] > 10000)
    ]

    print(f"Matching customers: {len(unusual_customers)}")

    if not unusual_customers.empty:
        print(
            unusual_customers[
                ["CustomerID"] + features
            ].head(10).round(2).to_string(index=False)
        )

    print("\n--- Feature Consistency Checks ---")

    print(
        "Customers with nonpositive monetary values:",
        int((df["Monetary"] <= 0).sum()),
    )

    print(
        "Customers with nonpositive quantities:",
        int((df["Total_Quantity"] <= 0).sum()),
    )

    print(
        "Customers with negative future revenue:",
        int((df["Future_Revenue"] < 0).sum()),
    )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()