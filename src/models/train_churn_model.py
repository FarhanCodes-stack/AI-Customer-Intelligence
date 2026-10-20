from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Observation_Days",
]

TARGET = "Churn"


def main():
    print("Loading churn dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    print(f"Customers: {len(df):,}")
    print(f"Features: {FEATURES}")
    print(f"Churn rate: {y.mean() * 100:.2f}%")

    # Hold out 20% of customers for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]),
    }

    results = []

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Active", "Churned"],
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "ROC_AUC": roc_auc,
        })

        # Save both models for later comparison and dashboard use.
        model_filename = (
            "logistic_churn_model.joblib"
            if name == "Logistic Regression"
            else "random_forest_churn_model.joblib"
        )

        joblib.dump(model, MODEL_DIR / model_filename)
        print(f"\nSaved model: {model_filename}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(results)
    print(results_df.round(4).to_string(index=False))

    print(
        "\nNote: these are preliminary results from a random "
        "customer-level split. Because customers share the same "
        "historical calendar windows, this evaluation does not "
        "establish how well the model will perform in a later "
        "calendar period."
    )


if __name__ == "__main__":
    main()