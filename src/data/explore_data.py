from pathlib import Path

import pandas as pd


# Find the project's root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the original dataset.
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


def main():
    # Check whether the dataset exists.
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        print("Please place Online Retail.xlsx inside data/raw/")
        return

    # Load the Excel dataset.
    print("Loading dataset... Please wait.")
    df = pd.read_excel(DATASET_PATH)

    # Display the first five rows.
    print("\n--- First Five Rows ---")
    print(df.head())

    # Display the dataset dimensions.
    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Display column names.
    print("\n--- Column Names ---")
    print(df.columns.tolist())

    # Display column data types and non-null counts.
    print("\n--- Dataset Information ---")
    df.info()

    # Display missing values in each column.
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

        # Check the number of unique customers.
    print("\n--- Unique Customers ---")
    print(df["CustomerID"].nunique())

    # Check the number of unique products.
    print("\n--- Unique Products ---")
    print(df["StockCode"].nunique())

    # Check the transaction date range.
    print("\n--- Transaction Date Range ---")
    print("Start Date:", df["InvoiceDate"].min())
    print("End Date:", df["InvoiceDate"].max())

    # Check for cancelled invoices.
    print("\n--- Cancelled Invoices ---")
    print(df["InvoiceNo"].astype(str).str.startswith("C").sum())

    # Check quantities and prices.
    print("\n--- Quantity and Price Statistics ---")
    print(df[["Quantity", "UnitPrice"]].describe())


if __name__ == "__main__":
    main()