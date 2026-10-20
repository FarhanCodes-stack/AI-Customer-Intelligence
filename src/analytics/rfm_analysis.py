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
    / "customer_rfm.csv"
)


def main():
    print("Loading cleaned transaction data...")

    df = pd.read_csv(DATA_PATH)

    # Convert InvoiceDate into datetime format.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Use the day after the final transaction as the reference date.
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    print(f"Reference date: {reference_date.date()}")

    # Calculate RFM metrics for each customer.
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda dates:
                     (reference_date - dates.max().normalize()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    # Save the customer-level RFM dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT_PATH, index=False)

    print("\n--- RFM Dataset Preview ---")
    print(rfm.head(10))

    print("\n--- RFM Dataset Information ---")
    print(f"Total customers: {len(rfm)}")
    print(f"Average recency: {rfm['Recency'].mean():.2f} days")
    print(f"Average frequency: {rfm['Frequency'].mean():.2f} orders")
    print(f"Average monetary value: £{rfm['Monetary'].mean():.2f}")

    print("\n--- RFM Statistics ---")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))

    print("\n--- Saved File ---")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()