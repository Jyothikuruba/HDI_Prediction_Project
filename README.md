# Human Development Index (HDI) Prediction Project

A statistical + machine learning project that predicts a country's **Human
Development Index (HDI)** tier — **Very High, High, Medium, or Low** — from
four core indicators:

- **Life expectancy at birth**
- **Mean years of schooling**
- **Expected years of schooling**
- **GNI per capita (PPP $)**

The project combines two complementary approaches:

1. **The official UNDP HDI formula** (`src/hdi_calculator.py`) — an exact,
   transparent, deterministic calculation used every year in the UN Human
   Development Report.
2. **A trained machine learning classifier** (`RandomForestClassifier`) that
   learns to predict the HDI tier directly from the raw indicators — useful
   as the "prediction model" referenced in the project scenarios.

Both are shown side by side in the CLI tool so predictions stay transparent
and explainable, not just a black-box output.

## How HDI is Calculated

```
Life Expectancy Index (LEI) = (LE - 20) / (85 - 20)

Mean Years Schooling Index   = MYS / 15
Expected Years Schooling Idx = EYS / 18
Education Index (EI)         = average of the two above

Income Index (II) = (ln(GNIpc) - ln(100)) / (ln(75000) - ln(100))

HDI = (LEI × EI × II) ^ (1/3)      [geometric mean]
```

**Classification tiers (UNDP standard cutoffs):**

| Tier       | HDI Range        |
|------------|------------------|
| Very High  | ≥ 0.800          |
| High       | 0.700 – 0.799    |
| Medium     | 0.550 – 0.699    |
| Low        | < 0.550          |

## Project Structure

```
hdi_project/
├── data/
│   └── synthetic_hdi_data.csv     # Generated training data (4,000 synthetic countries)
├── models/
│   └── hdi_model.pkl              # Trained RandomForestClassifier
├── src/
│   ├── hdi_calculator.py          # Official UNDP HDI formula implementation
│   ├── generate_data.py           # Synthetic dataset generator
│   ├── train_model.py             # Trains + evaluates the ML classifier
│   └── predict.py                 # CLI tool: predict HDI tier for any country
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
pip install -r requirements.txt

# 1. Generate the synthetic training dataset
python src/generate_data.py

# 2. Train the classifier (prints accuracy + feature importances)
python src/train_model.py

# 3. Predict for a custom country
python src/predict.py \
  --life_expectancy 82.5 \
  --mean_years_schooling 12.8 \
  --expected_years_schooling 16.5 \
  --gni_per_capita 52000

# Or run the three built-in project scenarios directly:
python src/predict.py
```

## The Three Project Scenarios

**Scenario 1 — Very High Human Development**
High life expectancy, strong schooling, high GNI per capita → model predicts
**Very High** HDI. Example: LE=83, MYS=13, EYS=17, GNI=$55,000 → `Very High`
(HDI ≈ 0.942).

**Scenario 2 — Emerging Economy / Development Gaps**
Mid-range indicators across the board → model predicts **Medium** HDI,
highlighting where healthcare, education, or income gains would move the
needle most. Example: LE=68, MYS=7.5, EYS=11.5, GNI=$8,500 → `Medium`
(HDI ≈ 0.656).

**Scenario 3 — Countries Requiring Development Intervention**
Low life expectancy, limited education, low GNI per capita → model predicts
**Low** HDI, flagging the country for policy prioritization. Example: LE=55,
MYS=3.0, EYS=6.5, GNI=$1,200 → `Low` (HDI ≈ 0.384).

## Model Performance

On a held-out 20% test split of the synthetic dataset, the RandomForest
classifier achieves **~96% accuracy** across all four tiers. Feature
importance analysis shows **GNI per capita** is the single strongest
predictor, followed by life expectancy and schooling indicators — consistent
with how the HDI formula weights these dimensions.

## Notes on the Dataset

The training data is **synthetically generated** (`src/generate_data.py`) to
span realistic indicator ranges with correlations similar to real countries
(higher income countries tend to have higher life expectancy and schooling).
Each synthetic country's true HDI and tier are computed with the exact UNDP
formula, so the ML model is learning to approximate that formula from raw
indicators — you can swap in real World Bank / UNDP country data with the
same four columns to retrain on actual country statistics.

## Extending This Project

- Swap synthetic data for real UNDP Human Development Report data (available
  at hdr.undp.org) to train on actual country statistics.
- Add a web front-end (Flask/Streamlit) around `predict.py` for interactive
  use by policymakers or researchers.
- Add regression output (predicted exact HDI score) alongside classification.
- Add historical trend analysis if multi-year data is used.
