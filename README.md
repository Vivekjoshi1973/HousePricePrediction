# House Price Prediction

A machine learning project that predicts house sale prices from property features. It includes an exploratory training notebook, a saved Random Forest model, and a Streamlit web app for making predictions.

**Live Demo:** [https://housepriceprediction-v.streamlit.app/](https://housepriceprediction-v.streamlit.app/)

## Project Structure

```
├── Notebook/
│   └── HousePricePrediction.ipynb   # EDA, training, evaluation, model export
├── Streamlit_App/
│   └── app.py                       # Web app for predictions
├── models/
│   ├── Random_Forest_House_Price.pkl  # Trained Random Forest model
│   └── columns.pkl                    # Expected feature columns
├── dataset/
│   └── house_price_50k.csv           # Training dataset (50,000 rows)
├── requirements.txt
└── README.md
```

## Dataset

`dataset/house_price_50k.csv` — 50,000 house listings with sale price and property attributes.

Raw columns: `ID`, `Date`, `Price`, `Bedrooms`, `Bathrooms`, `Sqft_living`, `Sqft_lot`, `Floors`, `Waterfront`, `View`, `Condition`, `Grade`, `Sqft_above`, `Sqft_basement`, `Yr_built`, `Yr_renovated`, `zipcode`, `Lat`, `Long`, `Sqft_living15`, `Sqft_lot15`.

## Pipeline

1. **Cleaning** — drop duplicates and unused columns (`ID`, `Date`, `Condition`, `zipcode`, `Lat`, `Long`, `Sqft_above`, `Sqft_basement`).
2. **Feature engineering**
   - `Is_renovated` = 1 if `Yr_renovated != 0`, else 0 (then drop `Yr_renovated`).
   - `Age` = 2026 − `Yr_built` (then drop `Yr_built`).
   - Cast `Bathrooms` and `Floors` to integers.
3. **EDA** — correlation heatmaps, price distribution, bedroom/bathroom/floor distributions.
4. **Scaling** — `StandardScaler` applied only to the feature set used by Linear Regression.
5. **Split** — 80/20 `train_test_split`, `random_state=42`.
6. **Train & evaluate** — Linear Regression, Decision Tree, Random Forest; export the best model.

## Features Used

Bedrooms, Bathrooms, Sqft_living, Sqft_lot, Floors, Waterfront, View, Grade, Sqft_living15, Sqft_lot15, Is_renovated, Age.

## Model Evaluation / Report of Every Model

This is a **regression** task (predicting a continuous sale price), so there is no sklearn
`classification_report`. The equivalent performance report uses **R² score (accuracy of the model,
in %)** together with MSE, RMSE and MAE. All models were evaluated on the same 20% hold-out test set.

| # | Model | R² Score (Accuracy) | MSE | RMSE | MAE |
|---|-------|--------------------:|----:|-----:|----:|
| 1 | Linear Regression | 64.72 % | 42,209,918,114.55 | $205,450.52 | $137,376.43 |
| 2 | Decision Tree Regressor (`max_depth=10`) | 62.83 % | 44,474,902,163.86 | $210,890.74 | $129,046.09 |
| 3 | **Random Forest Regressor (`n_estimators=200`)** | **73.92 %** | **31,199,597,831.93** | **$176,634.08** | **$114,421.44** |

### Which model has the highest accuracy, and why we chose it

**Random Forest Regressor — 73.92 % R² — is the best model and the one deployed.**

Reasons:

- **Highest accuracy:** 73.92 % of the variance in house prices is explained, versus 64.72 % for
  Linear Regression and 62.83 % for Decision Tree — an improvement of ~9 points over the next best.
- **Lowest errors:** it has the smallest RMSE ($176,634 vs $205,450 / $210,890) and the smallest MAE
  ($114,421 vs $137,376 / $129,046), so its typical prediction misses by the least amount of money.
- **Handles the non-linear data:** price relationships with `Sqft_living`, `Grade`, `Waterfront` and
  `Age` are non-linear and interactive. Linear Regression underfits them, while the single Decision
  Tree overfits and generalizes poorly (its R² is even lower than Linear Regression).
- **Bagging reduces variance:** Random Forest averages 200 decorrelated trees, which smooths out the
  noise and outliers present in a 50k-row price dataset while keeping the non-linear fit.
- **No scaling needed:** as a tree-based model it is invariant to feature scale, which simplifies the
  production pipeline (the Streamlit app feeds raw values straight into `model.predict`).

Because of this, `Random_Forest_House_Price.pkl` is exported with `joblib` and is the model loaded by
the web app.

## Setup

```bash
git clone https://github.com/Vivekjoshi1973/HousePricePrediction.git
cd HousePricePrediction
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook Notebook/HousePricePrediction.ipynb
```

The notebook performs EDA, trains Linear Regression, Decision Tree, and Random Forest models,
evaluates them (R² / MSE / RMSE / MAE), plots the R² comparison bar chart, and exports the best
model (Random Forest) to `models/`.

## Run the Streamlit App

```bash
streamlit run Streamlit_App/app.py
```

Then open the local URL shown in the terminal (default `http://localhost:8501`), fill in the house
details, and click **Predict Price**.

Or use the hosted app directly: [https://housepriceprediction-v.streamlit.app/](https://housepriceprediction-v.streamlit.app/)

### App inputs

Size & Layout: Bedrooms, Bathrooms, Floors, Living area, Lot size.
Quality & Condition: Build quality (1–13), View rating (0–4), Age, Waterfront, Renovated.
Neighborhood: Avg living area of 15 nearest neighbors, Avg lot size of 15 nearest neighbors.

The app validates the living area (minimum 200 sq.ft), shows the estimated price plus price per
sq.ft, and lets you review the submitted feature values.

## Model

- **Algorithm:** Random Forest Regressor (`n_estimators=200`, `random_state=42`)
- **Accuracy:** 73.92 % R² (best of the three models tested)
- **Artifact:** `models/Random_Forest_House_Price.pkl` (loaded with `joblib`)
- **Feature schema:** `models/columns.pkl`

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`, `jupyter`, `notebook`
(pinned in `requirements.txt`).
