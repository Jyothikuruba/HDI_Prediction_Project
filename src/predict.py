"""
predict.py
----------
Command-line tool that predicts a country's HDI tier from its four core
indicators, using BOTH:
    1. The trained ML model (models/hdi_model.pkl)
    2. The exact UNDP HDI formula (for comparison / transparency)

Usage:
    python src/predict.py --life_expectancy 82.5 --mean_years_schooling 12.8 \
        --expected_years_schooling 16.5 --gni_per_capita 52000

If no arguments are given, it walks through the three example scenarios from
the project brief (Very High / Medium / Low).
"""

import argparse
import os
import sys
import joblib

sys.path.append(os.path.dirname(__file__))
from hdi_calculator import compute_hdi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hdi_model.pkl")

FEATURES = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python src/generate_data.py` then "
            f"`python src/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_one(model, life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita):
    features = [[life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita]]
    ml_prediction = model.predict(features)[0]
    ml_proba = dict(zip(model.classes_, model.predict_proba(features)[0]))

    formula_result = compute_hdi(
        life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita
    )

    print("-" * 60)
    print("Inputs:")
    print(f"  Life expectancy:            {life_expectancy}")
    print(f"  Mean years of schooling:    {mean_years_schooling}")
    print(f"  Expected years of schooling:{expected_years_schooling}")
    print(f"  GNI per capita (PPP $):     {gni_per_capita}")
    print()
    print(f"ML Model Prediction:      {ml_prediction}")
    print("  Class probabilities:")
    for cls, p in sorted(ml_proba.items(), key=lambda x: -x[1]):
        print(f"    {cls:10s} {p:.3f}")
    print()
    print(f"Official HDI Formula:     {formula_result['hdi']}  -> {formula_result['category']}")
    print(f"  (Life Exp Index: {formula_result['life_expectancy_index']}, "
          f"Education Index: {formula_result['education_index']}, "
          f"Income Index: {formula_result['income_index']})")
    print("-" * 60)


def run_scenarios(model):
    print("\n=== Scenario 1: Very High Human Development ===")
    predict_one(model, life_expectancy=83.0, mean_years_schooling=13.0,
                expected_years_schooling=17.0, gni_per_capita=55000)

    print("\n=== Scenario 2: Emerging Economy (Medium) ===")
    predict_one(model, life_expectancy=68.0, mean_years_schooling=7.5,
                expected_years_schooling=11.5, gni_per_capita=8500)

    print("\n=== Scenario 3: Development Intervention Needed (Low) ===")
    predict_one(model, life_expectancy=55.0, mean_years_schooling=3.0,
                expected_years_schooling=6.5, gni_per_capita=1200)


def main():
    parser = argparse.ArgumentParser(description="Predict HDI category for a country.")
    parser.add_argument("--life_expectancy", type=float)
    parser.add_argument("--mean_years_schooling", type=float)
    parser.add_argument("--expected_years_schooling", type=float)
    parser.add_argument("--gni_per_capita", type=float)
    args = parser.parse_args()

    model = load_model()

    if None in (args.life_expectancy, args.mean_years_schooling,
                args.expected_years_schooling, args.gni_per_capita):
        run_scenarios(model)
    else:
        predict_one(model, args.life_expectancy, args.mean_years_schooling,
                    args.expected_years_schooling, args.gni_per_capita)


if __name__ == "__main__":
    main()
