import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from datetime import date
from xgboost import XGBClassifier

st.set_page_config(page_title="Rain Tomorrow Predictor (Sri Lanka)", layout="centered")

@st.cache_resource
def load_model_and_meta():
    with open("rain_artifact_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    m = XGBClassifier()
    m.load_model("xgb_booster.json")
    return m, meta

model, meta = load_model_and_meta()
THRESHOLD = float(meta["threshold"])
FEATURE_COLS = list(meta["feature_columns"])

DEFAULTS = {
    "shortwave_radiation_sum": 18.0,
    "et0_fao_evapotranspiration": 4.0,

    "windgusts_10m_max": 30.0,
    "winddirection_10m_dominant": 180.0,

    "apparent_temperature_max": 34.0,
    "apparent_temperature_min": 25.0,
    "apparent_temperature_mean": 29.5,

    "latitude": 7.0,
    "longitude": 80.0,
    "elevation": 50.0,

    "rain_today_0p1": 0,
}

st.title("🌧️ Rain Tomorrow Predictor (Sri Lanka)")
st.write("Enter basic weather details. The model predicts whether it will rain tomorrow")

city_cols = [c for c in FEATURE_COLS if c.startswith("city_")]
city_names = ["(Choose the city)"] + [c.replace("city_", "") for c in city_cols]
city_choice = st.selectbox("City", city_names, index=0)

st.subheader("Date")
chosen_date = st.date_input("Select today's date", value=date(2022, 6, 1))
month = chosen_date.month
day_of_year = chosen_date.timetuple().tm_yday
year = chosen_date.year

st.subheader("Input the Weather Details")

col1, col2 = st.columns(2)
with col1:
    tmax = st.number_input("Temperature Max (°C)", value=30.0)
    tmin = st.number_input("Temperature Min (°C)", value=23.0)
    tmean = st.number_input("Temperature Mean (°C)", value=26.0)

with col2:
    windspeed = st.number_input("Windspeed (km/h)", value=15.0)
    rain_today_0p1 = st.selectbox("Did it drizzle today? (≥ 0.1mm)", [0, 1], index=0)

def build_row():
    row = {c: 0 for c in FEATURE_COLS}

    for k, v in DEFAULTS.items():
        if k in row:
            row[k] = v

    user_values = {
        "temperature_2m_max": tmax,
        "temperature_2m_min": tmin,
        "temperature_2m_mean": tmean,
        "windspeed_10m_max": windspeed,
        "month": month,
        "day_of_year": day_of_year,
        "year": year,
        "rain_today_0p1": rain_today_0p1,
    }

    for k, v in user_values.items():
        if k in row:
            row[k] = v

    if city_choice != "(Choose the city)":
        col_name = "city_" + city_choice
        if col_name in row:
            row[col_name] = 1

    X = pd.DataFrame([row], columns=FEATURE_COLS)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return X

X_input = build_row()

if st.button("Predict"):
    prob = float(model.predict_proba(X_input)[0, 1])
    pred = 1 if prob >= THRESHOLD else 0

    st.subheader("Prediction")
    st.write(f"**Rain probability:** {prob:.3f}")
    st.write(f"**Decision (threshold={THRESHOLD:.2f}):** {'🌧️ Rain' if pred==1 else '☀️ No Rain'}")
    st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Explanation (SHAP)")
    st.caption("SHAP may take a few seconds.")

    background = meta.get("background", None)
    if background is None:
        st.warning("No background data found in meta. Save a small X_train sample as meta['background'] for SHAP.")
    else:
        background = background.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

        def predict_proba_fn(X):
            X_df = pd.DataFrame(X, columns=FEATURE_COLS)
            X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
            return model.predict_proba(X_df)[:, 1]

        @st.cache_resource
        def get_kernel_explainer(_background_np):
            return shap.KernelExplainer(predict_proba_fn, _background_np)

        explainer = get_kernel_explainer(background.values)
        shap_vals = explainer.shap_values(X_input.values, nsamples=200)

        shap_values_row = shap_vals[0]
        imp = pd.Series(shap_values_row, index=FEATURE_COLS).abs().sort_values(ascending=False).head(10)

        st.write("Top 10 feature contributions (absolute SHAP):")
        st.bar_chart(imp)