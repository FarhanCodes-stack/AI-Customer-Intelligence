from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_retail_data.csv"
)


def main():
    print("Loading cleaned customer transaction data...")

    df = pd.read_csv(DATA_PATH)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print("\n--- Dataset Overview ---")
    print(f"Total transactions: {len(df):,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"First transaction date: {df['InvoiceDate'].min()}")
    print(f"Last transaction date: {df['InvoiceDate'].max()}")

    print("\n--- Monthly Customer Activity ---")

    df["Month"] = df["InvoiceDate"].dt.to_period("M")

    monthly_activity = df.groupby("Month").agg(
        Transactions=("InvoiceNo", "nunique"),
        Active_Customers=("CustomerID", "nunique"),
        Revenue=("Revenue", "sum"),
    )

    monthly_activity["Revenue"] = (
        monthly_activity["Revenue"].round(2)
    )

    print(monthly_activity.to_string())

    print("\n--- Customer Last Purchase Distribution ---")

    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()

    reference_date = df["InvoiceDate"].max().normalize()

    days_since_purchase = (
        reference_date - last_purchase.dt.normalize()
    ).dt.days

    print("\nDays since last purchase:")
    print(days_since_purchase.describe().round(2).to_string())

    print("\nCustomers by inactivity period:")

    inactivity_groups = {
        "0-30 days": (
            (days_since_purchase >= 0)
            & (days_since_purchase <= 30)
        ),
        "31-60 days": (
            (days_since_purchase > 30)
            & (days_since_purchase <= 60)
        ),
        "61-90 days": (
            (days_since_purchase > 60)
            & (days_since_purchase <= 90)
        ),
        "91-180 days": (
            (days_since_purchase > 90)
            & (days_since_purchase <= 180)
        ),
        "181+ days": days_since_purchase > 180,
    }

    for label, condition in inactivity_groups.items():
        count = int(condition.sum())
        percentage = count / len(days_since_purchase) * 100

        print(
            f"{label}: {count:,} customers "
            f"({percentage:.2f}%)"
        )

    print("\nChurn data inspection complete.")


if __name__ == "__main__":
    main()