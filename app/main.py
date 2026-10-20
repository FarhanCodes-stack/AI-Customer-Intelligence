from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PATH = PROCESSED_DIR / "cleaned_retail_data.csv"
SEGMENTS_PATH = PROCESSED_DIR / "customer_segments.csv"
RFM_PATH = PROCESSED_DIR / "customer_rfm.csv"
CHURN_DATA_PATH = PROCESSED_DIR / "customer_churn_data.csv"
VALUE_DATA_PATH = PROCESSED_DIR / "customer_value_data.csv"
VALUE_REPORT_PATH = (
    PROCESSED_DIR / "customer_value_prediction_report.csv"
)

CHURN_MODEL_PATH = PROCESSED_DIR / "logistic_churn_model.joblib"
PURCHASE_MODEL_PATH = (
    PROCESSED_DIR / "customer_purchase_classifier.joblib"
)
REVENUE_MODEL_PATH = (
    PROCESSED_DIR / "customer_positive_revenue_model.joblib"
)


st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. Data loading
# --------------------------------------------------

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def money(value):
    return f"£{value:,.2f}"


def show_missing_file_message(path):
    st.warning(
        f"Required file not found: {path.name}. "
        "Run the corresponding data preparation or training script."
    )


# --------------------------------------------------
# 3. Validate the main dataset
# --------------------------------------------------

st.title("📊 AI-Powered Customer Intelligence Platform")

st.write(
    "Analyze customer behaviour, understand churn risk, "
    "explore customer segments, and estimate future customer revenue."
)

if not DATA_PATH.exists():
    st.error(
        "Cleaned dataset not found. "
        "Run src/data/clean_data.py first."
    )
    st.stop()

df = load_data(DATA_PATH)


# --------------------------------------------------
# 4. Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Overview",
        "Customer Segmentation",
        "Churn Analysis",
        "Customer Value Prediction",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset: UCI Online Retail")
st.sidebar.caption(f"Transaction records: {len(df):,}")


# ==================================================
# PAGE 1: BUSINESS OVERVIEW
# ==================================================

if page == "Business Overview":

    total_revenue = df["Revenue"].sum()
    total_customers = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()
    average_order_value = (
        total_revenue / total_orders if total_orders else 0
    )

    st.subheader("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", money(total_revenue))
    col2.metric("Unique Customers", f"{total_customers:,}")
    col3.metric("Total Orders", f"{total_orders:,}")
    col4.metric("Average Order Value", money(average_order_value))

    monthly_revenue = (
        df.assign(Month=df["InvoiceDate"].dt.to_period("M").astype(str))
        .groupby("Month")["Revenue"]
        .sum()
        .sort_index()
    )

    country_revenue = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Monthly Revenue")
        st.line_chart(monthly_revenue)

    with right_col:
        st.subheader("Top 10 Countries by Revenue")
        st.bar_chart(country_revenue)

    st.divider()
    st.subheader("Recent Transactions")

    recent_transactions = (
        df.sort_values("InvoiceDate", ascending=False)
        .head(20)
    )

    st.dataframe(
        recent_transactions,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Source: UCI Online Retail. Revenue represents retained "
        "transaction lines after cleaning; cancellations and "
        "nonpositive quantity or price records were excluded."
    )


# ==================================================
# PAGE 2: CUSTOMER SEGMENTATION
# ==================================================

elif page == "Customer Segmentation":

    st.header("👥 Customer Segmentation")

    if not SEGMENTS_PATH.exists():
        show_missing_file_message(SEGMENTS_PATH)
        st.code("python src/models/customer_segmentation.py")
        st.stop()

    segments = load_data(SEGMENTS_PATH)

    required_columns = {
        "CustomerID",
        "Cluster",
        "Recency",
        "Frequency",
        "Monetary",
    }

    if not required_columns.issubset(segments.columns):
        st.error(
            "The segmentation file is missing required columns. "
            "Check data/processed/customer_segments.csv."
        )
        st.stop()

    segments["CustomerID"] = segments["CustomerID"].astype(str)

    segment_summary = (
        segments.groupby("Cluster", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Spending=("Monetary", "mean"),
            Total_Spending=("Monetary", "sum"),
        )
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers Segmented", f"{segments['CustomerID'].nunique():,}")
    col2.metric("Segments Identified", f"{segments['Cluster'].nunique()}")
    col3.metric("Average Historical Spending", money(segments["Monetary"].mean()))

    st.subheader("Segment Profiles")

    st.dataframe(
        segment_summary.style.format({
            "Avg_Recency": "{:.1f} days",
            "Avg_Frequency": "{:.2f}",
            "Avg_Spending": "£{:,.2f}",
            "Total_Spending": "£{:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Customers per Segment")
        st.bar_chart(
            segment_summary.set_index("Cluster")["Customers"]
        )

    with right_col:
        st.subheader("Average Spending per Segment")
        st.bar_chart(
            segment_summary.set_index("Cluster")["Avg_Spending"]
        )

    st.subheader("Explore Customers")

    selected_cluster = st.selectbox(
        "Choose a customer segment",
        sorted(segments["Cluster"].unique()),
    )

    selected_customers = segments[
        segments["Cluster"] == selected_cluster
    ].sort_values("Monetary", ascending=False)

    st.dataframe(
        selected_customers,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Clusters are machine-generated groups based on RFM behaviour. "
        "Cluster numbers are identifiers, not rankings. Interpret each "
        "group using its recency, frequency, and spending profile."
    )


# ==================================================
# PAGE 3: CHURN ANALYSIS
# ==================================================

elif page == "Churn Analysis":

    st.header("⚠️ Customer Churn Analysis")

    if not CHURN_DATA_PATH.exists():
        show_missing_file_message(CHURN_DATA_PATH)
        st.code("python src/models/prepare_churn_data.py")
        st.stop()

    if not CHURN_MODEL_PATH.exists():
        show_missing_file_message(CHURN_MODEL_PATH)
        st.code("python src/models/train_churn_model.py")
        st.stop()

    churn_data = load_data(CHURN_DATA_PATH)
    churn_model = load_model(CHURN_MODEL_PATH)

    features = list(churn_model.feature_names_in_)

    missing_columns = [
        column for column in features
        if column not in churn_data.columns
    ]

    if missing_columns:
        st.error(
            "Churn data is missing columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    X_churn = churn_data[features]

    churn_probability = churn_model.predict_proba(X_churn)[:, 1]

    churn_results = churn_data.copy()
    churn_results["Predicted_Churn_Probability"] = churn_probability
    churn_results["Predicted_Churn_Risk"] = np.where(
        churn_probability >= 0.5,
        "Higher",
        "Lower",
    )

    churn_rate = (
        churn_data["Churn"].mean() * 100
        if "Churn" in churn_data.columns
        else np.nan
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers Analyzed", f"{len(churn_results):,}")
    col2.metric(
        "Historical Churn Rate",
        f"{churn_rate:.1f}%" if not np.isnan(churn_rate) else "N/A",
    )
    col3.metric(
        "Customers with Higher Predicted Risk",
        f"{(churn_results['Predicted_Churn_Risk'] == 'Higher').sum():,}",
    )

    st.warning(
        "This model uses historical transactions. Its churn label means "
        "no purchase during a defined three-month prediction window; "
        "it does not prove permanent churn or current real-world risk."
    )

    st.subheader("Predicted Churn Risk Distribution")

    risk_counts = (
        churn_results["Predicted_Churn_Risk"]
        .value_counts()
        .reindex(["Higher", "Lower"], fill_value=0)
    )

    st.bar_chart(risk_counts)

    st.subheader("Find a Customer")

    search_id = st.text_input(
        "Customer ID",
        placeholder="Enter a CustomerID",
    )

    risk_filter = st.selectbox(
        "Filter by predicted risk",
        ["All", "Higher", "Lower"],
    )

    filtered_churn = churn_results.copy()

    if search_id.strip():
        filtered_churn = filtered_churn[
            filtered_churn["CustomerID"].astype(str).str.contains(
                search_id.strip(),
                regex=False,
            )
        ]

    if risk_filter != "All":
        filtered_churn = filtered_churn[
            filtered_churn["Predicted_Churn_Risk"] == risk_filter
        ]

    display_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Predicted_Churn_Probability",
        "Predicted_Churn_Risk",
    ]

    st.dataframe(
        filtered_churn[display_columns].sort_values(
            "Predicted_Churn_Probability",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Churn Analysis CSV",
        data=churn_results.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
    )


# ==================================================
# PAGE 4: CUSTOMER VALUE PREDICTION
# ==================================================

elif page == "Customer Value Prediction":

    st.header("💰 Customer Future Value Prediction")

    if not VALUE_DATA_PATH.exists():
        show_missing_file_message(VALUE_DATA_PATH)
        st.code("python src/models/prepare_customer_value_data.py")
        st.stop()

    if not PURCHASE_MODEL_PATH.exists() or not REVENUE_MODEL_PATH.exists():
        st.warning(
            "The two-stage customer value models were not found. "
            "Train them before using this section."
        )
        st.code(
            "python src/models/train_two_stage_customer_value_model.py"
        )
        st.stop()

    value_data = load_data(VALUE_DATA_PATH)
    purchase_model = load_model(PURCHASE_MODEL_PATH)
    revenue_model = load_model(REVENUE_MODEL_PATH)

    features = list(purchase_model.feature_names_in_)

    missing_columns = [
        column for column in features
        if column not in value_data.columns
    ]

    if missing_columns:
        st.error(
            "Customer value data is missing columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    X_value = value_data[features]
    purchase_probability = purchase_model.predict_proba(X_value)[:, 1]

    conditional_revenue = np.expm1(
        revenue_model.predict(X_value)
    )

    expected_revenue = (
        purchase_probability * conditional_revenue
    )

    value_results = pd.DataFrame({
        "CustomerID": value_data["CustomerID"],
        "Recency": value_data["Recency"],
        "Frequency": value_data["Frequency"],
        "Historical_Revenue": value_data["Monetary"],
        "Purchase_Probability": purchase_probability,
        "Predicted_Revenue_If_Purchasing": conditional_revenue,
        "Expected_Future_Revenue": expected_revenue,
        "Actual_Future_Revenue": value_data["Future_Revenue"],
    })

    value_results["Customer_Priority"] = pd.cut(
        value_results["Expected_Future_Revenue"],
        bins=[-np.inf, 100, 500, 2000, np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    )

    value_results = value_results.sort_values(
        "Expected_Future_Revenue",
        ascending=False,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers Analyzed",
        f"{len(value_results):,}",
    )
    col2.metric(
        "Average Expected Future Revenue",
        money(value_results["Expected_Future_Revenue"].mean()),
    )
    col3.metric(
        "Very High Priority Customers",
        f"{(value_results['Customer_Priority'] == 'Very High').sum():,}",
    )

    st.info(
        "These are historical demonstration predictions. The report "
        "applies saved models to the dataset used in earlier experiments; "
        "it should not be interpreted as a validated forecast of current "
        "customer behaviour."
    )

    st.subheader("Expected Revenue by Priority")

    priority_revenue = (
        value_results.groupby(
            "Customer_Priority",
            observed=False,
        )["Expected_Future_Revenue"]
        .mean()
        .reindex(["Low", "Medium", "High", "Very High"])
    )

    st.bar_chart(priority_revenue)

    st.subheader("Search and Filter Customers")

    search_id = st.text_input(
        "Search Customer ID",
        placeholder="Enter a CustomerID",
        key="value_customer_search",
    )

    selected_priority = st.selectbox(
        "Customer priority",
        ["All", "Low", "Medium", "High", "Very High"],
    )

    filtered_values = value_results.copy()

    if search_id.strip():
        filtered_values = filtered_values[
            filtered_values["CustomerID"].astype(str).str.contains(
                search_id.strip(),
                regex=False,
            )
        ]

    if selected_priority != "All":
        filtered_values = filtered_values[
            filtered_values["Customer_Priority"] == selected_priority
        ]

    st.dataframe(
        filtered_values,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download Customer Value Predictions CSV",
        data=value_results.to_csv(index=False).encode("utf-8"),
        file_name="customer_value_predictions.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Powered Customer Intelligence Platform | "
    "Built with Python, Pandas, Scikit-learn, and Streamlit. "
    "Historical findings are illustrative and do not guarantee future outcomes."
)