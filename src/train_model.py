"""
train_model.py
---------------
Trains a RandomForestClassifier to predict HDI category (Very High / High /
Medium / Low) directly from the four raw indicators:
    - life_expectancy
    - mean_years_schooling
    - expected_years_schooling
    - gni_per_capita

This mirrors the project's use case: a user/policymaker/researcher enters
indicator values and the model predicts the development tier, without needing
to compute the HDI formula by hand.

Run:
    python src/generate_data.py   # creates data/synthetic_hdi_data.csv
    python src/train_model.py     # trains + saves model to models/hdi_model.pkl
"""

import os
import csv
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_hdi_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hdi_model.pkl")

FEATURES = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"]
TARGET = "hdi_category"


def load_data():
    X, y = [], []
    with open(DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[feat]) for feat in FEATURES])
            y.append(row[TARGET])
    return X, y


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generate_data.py` first."
        )

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:28s} {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
