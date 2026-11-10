# House Price Prediction

A machine learning project that predicts house sale prices from property features. It includes an exploratory training notebook, a saved Random Forest model, and a Streamlit web app for making predictions.

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
│   └── house_price_50k.csv           # Training dataset
└── requirements.txt
```

## Features Used

Bedrooms, Bathrooms, Sqft_living, Sqft_lot, Floors, Waterfront, View, Grade, Sqft_living15, Sqft_lot15, Is_renovated, Age.

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

The notebook performs EDA, trains Linear Regression, Decision Tree, and Random Forest models, evaluates them, and exports the best model (Random Forest) to `models/`.

## Run the Streamlit App

```bash
streamlit run Streamlit_App/app.py
```

Then open the local URL shown in the terminal (default `http://localhost:8501`), fill in the house details, and click **Predict Price**.

## Model

- **Algorithm:** Random Forest Regressor
- **Artifact:** `models/Random_Forest_House_Price.pkl` (loaded with `joblib`)
