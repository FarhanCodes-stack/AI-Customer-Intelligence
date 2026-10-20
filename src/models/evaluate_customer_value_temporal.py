import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

DATA_PATH = "data/processed/cleaned_retail_data.csv"
OUTPUT_PATH = "data/processed/customer_value_temporal_validation.csv"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]


def build_customer_dataset(df, observation_start, observation_end,
                           prediction_start, prediction_end):

    observation = df[
        (df["InvoiceDate"] >= observation_start)
        & (df["InvoiceDate"] <= observation_end)
    ].copy()

    prediction = df[
        (df["InvoiceDate"] >= prediction_start)
        & (df["InvoiceDate"] <= prediction_end)
    ].copy()

    if observation.empty:
        raise ValueError("Observation period contains no transactions.")

    if prediction.empty:
        raise ValueError("Prediction period contains no transactions.")

    observation_end_date = pd.Timestamp(observation_end).normalize()

    features = observation.groupby("CustomerID").agg(
        Last_Purchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Unique_Products=("StockCode", "nunique"),
    )

    features["Recency"] = (
        observation_end_date
        - features["Last_Purchase"].dt.normalize()
    ).dt.days

    features = features.drop(columns=["Last_Purchase"])

    targets = prediction.groupby("CustomerID")["Revenue"].sum()
    targets.name = "Future_Revenue"

    dataset = features.join(targets, how="left")
    dataset["Future_Revenue"] = dataset["Future_Revenue"].fillna(0)

    return dataset.reset_index()


def evaluate_temporal_validation():

    print("Loading cleaned transaction data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["InvoiceDate"]
    )

    # Training: observe December-March, predict April-June.
    train_data = build_customer_dataset(
        df,
        "2010-12-01",
        "2011-03-31 23:59:59",
        "2011-04-01",
        "2011-06-30 23:59:59",
    )

    # Testing: observe December-June, predict July-September.
    test_data = build_customer_dataset(
        df,
        "2010-12-01",
        "2011-06-30 23:59:59",
        "2011-07-01",
        "2011-09-30 23:59:59",
    )

    X_train = train_data[FEATURES]
    y_train = train_data["Future_Revenue"]

    X_test = test_data[FEATURES]
    y_test = test_data["Future_Revenue"]

    print(f"\nTraining customers: {len(train_data)}")
    print(f"Testing customers: {len(test_data)}")
    print(f"Training future buyers: {(y_train > 0).sum()}")
    print(f"Testing future buyers: {(y_test > 0).sum()}")

    # Stage 1: Predict whether a customer will purchase.
    purchase_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    purchase_model.fit(X_train, (y_train > 0).astype(int))

    purchase_probability = purchase_model.predict_proba(X_test)[:, 1]

    # Stage 2: Predict revenue among customers who actually purchased
    # in the training prediction period.
    positive_train = y_train > 0

    revenue_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    revenue_model.fit(
        X_train.loc[positive_train],
        np.log1p(y_train.loc[positive_train]),
    )

    conditional_revenue = np.expm1(
        revenue_model.predict(X_test)
    )

    # Expected revenue = purchase probability * revenue if purchasing.
    predictions = purchase_probability * conditional_revenue
    predictions = np.maximum(predictions, 0)

    # Baseline predicts the average revenue from the training period.
    baseline_predictions = np.full(
        len(y_test),
        y_train.mean(),
    )

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    try:
        auc = roc_auc_score(
            (y_test > 0).astype(int),
            purchase_probability,
        )
    except ValueError:
        auc = np.nan

    results = pd.DataFrame([
        {
            "Model": "Training-mean baseline",
            "Test_Customers": len(y_test),
            "Actual_Average_Revenue": y_test.mean(),
            "Predicted_Average_Revenue": baseline_predictions.mean(),
            "MAE": baseline_mae,
            "RMSE": baseline_rmse,
            "Purchase_ROC_AUC": np.nan,
        },
        {
            "Model": "Two-stage model",
            "Test_Customers": len(y_test),
            "Actual_Average_Revenue": y_test.mean(),
            "Predicted_Average_Revenue": predictions.mean(),
            "MAE": model_mae,
            "RMSE": model_rmse,
            "Purchase_ROC_AUC": auc,
        },
    ])

    results.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Time-Based Validation Results ---")
    print(results.round(3).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_PATH}")

    if model_mae < baseline_mae:
        print(
            "\nThe two-stage model has lower MAE than the baseline "
            "on this later time period."
        )
    else:
        print(
            "\nThe two-stage model did not beat the baseline on MAE "
            "in this later time period."
        )


if __name__ == "__main__":
    evaluate_temporal_validation()