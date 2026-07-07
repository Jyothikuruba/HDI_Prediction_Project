"""
hdi_calculator.py
-----------------
Implements the official UNDP Human Development Index (HDI) methodology.

HDI is the geometric mean of three normalized sub-indices:
    1. Life Expectancy Index (LEI)
    2. Education Index (EI)      -> mean of Mean Years of Schooling Index (MYSI)
                                     and Expected Years of Schooling Index (EYSI)
    3. Income Index (II)         -> based on GNI per capita (PPP $)

Standard UNDP goalposts (used every year in the Human Development Report):
    Life expectancy at birth:      20 - 85 years
    Mean years of schooling:        0 - 15 years
    Expected years of schooling:    0 - 18 years
    GNI per capita (PPP $):       100 - 75,000

Classification tiers (UNDP standard cutoffs):
    Very High : HDI >= 0.800
    High      : 0.700 <= HDI < 0.800
    Medium    : 0.550 <= HDI < 0.700
    Low       : HDI < 0.550
"""

import math

# ---- UNDP goalposts ----
LIFE_EXP_MIN, LIFE_EXP_MAX = 20, 85
MYS_MAX = 15          # Mean Years of Schooling max
EYS_MAX = 18          # Expected Years of Schooling max
GNI_MIN, GNI_MAX = 100, 75000


def life_expectancy_index(life_expectancy: float) -> float:
    value = (life_expectancy - LIFE_EXP_MIN) / (LIFE_EXP_MAX - LIFE_EXP_MIN)
    return min(max(value, 0), 1)


def education_index(mean_years_schooling: float, expected_years_schooling: float) -> float:
    mysi = min(max(mean_years_schooling / MYS_MAX, 0), 1)
    eysi = min(max(expected_years_schooling / EYS_MAX, 0), 1)
    return (mysi + eysi) / 2


def income_index(gni_per_capita: float) -> float:
    gni_clamped = min(max(gni_per_capita, GNI_MIN), GNI_MAX)
    value = (math.log(gni_clamped) - math.log(GNI_MIN)) / (math.log(GNI_MAX) - math.log(GNI_MIN))
    return min(max(value, 0), 1)


def compute_hdi(life_expectancy: float, mean_years_schooling: float,
                expected_years_schooling: float, gni_per_capita: float) -> dict:
    """Returns the sub-indices and final HDI score."""
    lei = life_expectancy_index(life_expectancy)
    ei = education_index(mean_years_schooling, expected_years_schooling)
    ii = income_index(gni_per_capita)

    hdi = (lei * ei * ii) ** (1 / 3)

    return {
        "life_expectancy_index": round(lei, 4),
        "education_index": round(ei, 4),
        "income_index": round(ii, 4),
        "hdi": round(hdi, 4),
        "category": classify_hdi(hdi),
    }


def classify_hdi(hdi: float) -> str:
    if hdi >= 0.800:
        return "Very High"
    elif hdi >= 0.700:
        return "High"
    elif hdi >= 0.550:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    # Quick sanity check
    result = compute_hdi(
        life_expectancy=82.5,
        mean_years_schooling=12.8,
        expected_years_schooling=16.5,
        gni_per_capita=52000,
    )
    print(result)
