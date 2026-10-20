from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_value_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
]

TARGET = "Future_Revenue"


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Revenue cannot be negative.
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"MAE: £{mae:,.2f}")
    print(f"RMSE: £{rmse:,.2f}")

    print("\nSample predictions:")
    comparison = pd.DataFrame({
        "Actual_Revenue": y_test.to_numpy()[:10],
        "Predicted_Revenue": predictions[:10],
    })

    print(comparison.round(2).to_string(index=False))

    return model, mae, rmse


def main():
    print("Loading customer value dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Total customers: {len(df):,}")
    print(f"Training customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")

    # Baseline: predict the average revenue in the training set.
    baseline_predictions = np.full(
        len(y_test),
        y_train.mean(),
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    print("\n" + "=" * 60)
    print("BASELINE: TRAINING-MEAN PREDICTION")
    print("=" * 60)
    print(f"MAE: £{baseline_mae:,.2f}")
    print(f"RMSE: £{baseline_rmse:,.2f}")

    models = {
        "Ridge Regression": TransformedTargetRegressor(
            regressor=Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        trained_model, mae, rmse = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        trained_models[name] = trained_model

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
        })

    results.append({
        "Model": "Training-Mean Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
    })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.round(2).to_string(index=False))

    # Save the model with the lowest test-set MAE.
    best_model_name = min(
        trained_models,
        key=lambda name: next(
            result["MAE"]
            for result in results
            if result["Model"] == name
        ),
    )

    best_model = trained_models[best_model_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "customer_value_model.joblib",
    )

    results_df.to_csv(
        MODEL_DIR / "customer_value_model_results.csv",
        index=False,
    )

    print(f"\nBest model by test MAE: {best_model_name}")
    print("Saved model: customer_value_model.joblib")
    print("Saved comparison: customer_value_model_results.csv")

    print(
        "\nNote: this is a preliminary random-split evaluation. "
        "The next validation improvement should use a later "
        "calendar period as the test set."
    )


if __name__ == "__main__":
    main()