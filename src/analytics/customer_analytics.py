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

    # Convert invoice dates into datetime values.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calculate overall business metrics.
    total_revenue = df["Revenue"].sum()
    total_customers = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()

    # Average order value = total revenue / number of orders.
    average_order_value = total_revenue / total_orders

    print("\n========== BUSINESS OVERVIEW ==========")
    print(f"Total Revenue: £{total_revenue:,.2f}")
    print(f"Unique Customers: {total_customers:,}")
    print(f"Unique Orders: {total_orders:,}")
    print(f"Average Order Value: £{average_order_value:,.2f}")

    # Calculate revenue by country.
    country_revenue = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== TOP 10 COUNTRIES BY REVENUE ==========")
    print(country_revenue.head(10).round(2))

    # Calculate monthly revenue.
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly_revenue = (
        df.groupby("Month")["Revenue"]
        .sum()
        .sort_index()
    )

    print("\n========== MONTHLY REVENUE ==========")
    print(monthly_revenue.round(2))


if __name__ == "__main__":
    main()