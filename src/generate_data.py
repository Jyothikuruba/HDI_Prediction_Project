"""
generate_data.py
----------------
Generates a synthetic dataset of "countries" with realistic-ranged development
indicators, computes their true HDI using the official formula (hdi_calculator.py),
and saves everything to data/synthetic_hdi_data.csv.

This dataset is what the ML model in train_model.py learns from: given the four
raw indicators, predict which of the four HDI tiers (Very High / High / Medium / Low)
a country falls into.
"""

import csv
import random
import os
import sys

sys.path.append(os.path.dirname(__file__))
from hdi_calculator import compute_hdi

random.seed(42)

N_SAMPLES = 4000
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_hdi_data.csv")


def sample_country():
    """Sample correlated-ish indicators so the data resembles real countries
    (e.g., higher GNI tends to come with higher life expectancy & schooling)."""
    development_level = random.random()  # 0 = poor, 1 = rich, underlying latent factor

    life_expectancy = random.gauss(50 + development_level * 35, 4)
    life_expectancy = min(max(life_expectancy, 45), 86)

    mean_years_schooling = random.gauss(2 + development_level * 11, 1.5)
    mean_years_schooling = min(max(mean_years_schooling, 0.5), 14.5)

    expected_years_schooling = random.gauss(6 + development_level * 11, 1.8)
    expected_years_schooling = min(max(expected_years_schooling, 3), 18)

    # income tends to scale exponentially with development level
    gni_per_capita = math_exp_income(development_level)

    return life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita


def math_exp_income(development_level):
    import math
    base = 400
    scale = 90000
    noise = random.gauss(0, 0.15)
    value = base * math.exp((math.log(scale / base)) * development_level + noise)
    return min(max(value, 300), 95000)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "life_expectancy",
            "mean_years_schooling",
            "expected_years_schooling",
            "gni_per_capita",
            "hdi_score",
            "hdi_category",
        ])

        for _ in range(N_SAMPLES):
            le, mys, eys, gni = sample_country()
            result = compute_hdi(le, mys, eys, gni)
            writer.writerow([
                round(le, 2),
                round(mys, 2),
                round(eys, 2),
                round(gni, 2),
                result["hdi"],
                result["category"],
            ])

    print(f"Wrote {N_SAMPLES} synthetic country records to {OUT_PATH}")


if __name__ == "__main__":
    main()
