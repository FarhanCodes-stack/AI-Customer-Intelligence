from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_retail_data.csv"
)


def main():
    print("Loading original dataset...")

    df = pd.read_excel(RAW_DATA_PATH)

    original_rows = len(df)
    print(f"Original rows: {original_rows}")

    # Remove rows without a customer ID.
    df = df.dropna(subset=["CustomerID"])

    # Remove rows without a product description.
    df = df.dropna(subset=["Description"])

    # Exclude cancelled invoices.
    df = df[
        ~df["InvoiceNo"].astype(str).str.startswith("C")
    ]

    # Keep only positive quantities and unit prices.
    df = df[
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ]

    # Convert CustomerID to an integer.
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Calculate revenue for each transaction line.
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Create the output directory if it does not exist.
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the cleaned dataset.
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\n--- Cleaning Results ---")
    print(f"Cleaned rows: {len(df)}")
    print(f"Rows removed: {original_rows - len(df)}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique products: {df['StockCode'].nunique()}")
    print(f"Total revenue: {df['Revenue'].sum():.2f}")

    print("\n--- Saved File ---")
    print(PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()