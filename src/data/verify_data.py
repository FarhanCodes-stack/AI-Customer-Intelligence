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
    print("Loading cleaned dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- First Five Rows ---")
    print(df.head())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Revenue Verification ---")
    print(f"Total revenue: {df['Revenue'].sum():.2f}")

    print("\n--- Validation Checks ---")
    print("Missing CustomerIDs:", df["CustomerID"].isnull().sum())
    print("Missing descriptions:", df["Description"].isnull().sum())
    print("Non-positive quantities:", (df["Quantity"] <= 0).sum())
    print("Non-positive prices:", (df["UnitPrice"] <= 0).sum())
    print(
        "Cancelled invoice rows:",
        df["InvoiceNo"].astype(str).str.startswith("C").sum(),
    )


if __name__ == "__main__":
    main()