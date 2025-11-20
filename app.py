import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Load data & model
# -----------------------------

@st.cache_data
def load_data():
    # This CSV must be in the same folder as app.py
    df = pd.read_csv("cardekho_data.csv")

    # Remove outliers (same logic as during training)
    numeric_cols = ["Selling_Price", "Present_Price", "Kms_Driven", "Year"]
    clean_df = df.copy()
    for col in numeric_cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]

    # Feature engineering
    clean_df["Car_Age"] = 2019 - clean_df["Year"]
    clean_df["Brand"] = clean_df["Car_Name"].apply(lambda x: x.split()[0].lower())

    # One-hot encode categorical variables
    clean_df = pd.get_dummies(
        clean_df,
        columns=["Brand", "Fuel_Type", "Seller_Type", "Transmission"],
        drop_first=True
    )

    # Drop unused columns
    clean_df = clean_df.drop(["Car_Name", "Year"], axis=1)

    return clean_df


@st.cache_resource
def load_model():
    # This model file must also be in the same folder as app.py
    model = joblib.load("car_price_rf_model.joblib")
    return model


data = load_data()
model = load_model()

# Feature columns (all except target)
FEATURE_COLUMNS = [c for c in data.columns if c != "Selling_Price"]

# Extract brands from dummy

