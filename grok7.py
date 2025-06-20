import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# -------------------- Load Model + Preprocessors --------------------
try:
    model = joblib.load('random_forest_model111.pkl')
    ohe = joblib.load('one_hot_encoder111.pkl')
    scaler = joblib.load('standard_scaler111.pkl')
    feature_names = joblib.load('feature_names111.pkl')
except FileNotFoundError:
    st.error("❌ Required files not found. Make sure all required `.pkl` files are present.")
    st.stop()

# -------------------- Preprocessing --------------------
def preprocess_input(data, ohe, scaler, feature_names):
    input_df = pd.DataFrame([data])
    input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'], errors='coerce')
    input_df['hour'] = input_df['trans_date_trans_time'].dt.hour
    input_df['day_of_week'] = input_df['trans_date_trans_time'].dt.dayofweek
    input_df['day'] = input_df['trans_date_trans_time'].dt.day
    input_df['month'] = input_df['trans_date_trans_time'].dt.month
    input_df['is_night'] = input_df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    # Use provided distance directly
    input_df['location_diff_km'] = input_df['distance_from_last_txn_km']
    input_df['velocity_kmph'] = input_df['location_diff_km'] / (input_df['time_diff'] / 3600)
    input_df['velocity_kmph'].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df['velocity_kmph'].fillna(0, inplace=True)
    input_df['unrealistic_speed_flag'] = (input_df['velocity_kmph'] > 1000).astype(int)
    input_df['suspicious_distance_jump'] = ((input_df['location_diff_km'] >= 1000) & (input_df['time_diff'] <= 60)).astype(int)

    try:
        input_scaled = scaler.transform(input_df[scaler.feature_names_in_])
        input_scaled_df = pd.DataFrame(input_scaled, columns=scaler.feature_names_in_, index=input_df.index)

        input_ohe = ohe.transform(input_df[ohe.feature_names_in_])
        encoded_cols = ohe.get_feature_names_out(ohe.feature_names_in_)
        input_ohe_df = pd.DataFrame(input_ohe, columns=encoded_cols, index=input_df.index)

        final_input = pd.concat([input_scaled_df, input_ohe_df], axis=1)
        final_input = final_input.reindex(columns=feature_names, fill_value=0)
        return final_input
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

# -------------------- Title --------------------
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("Predict whether a credit card transaction is fraudulent using a trained Random Forest model.")

# -------------------- Input Form --------------------
st.header("📝 Enter Transaction Details")

categorical_inputs = {}
with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        trans_date = st.date_input("Transaction Date", value=datetime.date.today())
        trans_time = st.time_input("Transaction Time", value=datetime.time(12, 0))
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
        login_attempts = st.number_input("Login Attempts", min_value=0, value=1, step=1)
        time_diff = st.number_input("Time Since Last Transaction (seconds)", min_value=0.0, value=0.0, step=0.1)
        txn_last_24h = st.number_input("Transactions in Last 24 Hours", min_value=0, value=1, step=1)
        account_age_days = st.number_input("Account Age (Days)", min_value=0, value=30, step=1)
        lat = st.number_input("Current Latitude", value=40.0, step=0.01)
        long = st.number_input("Current Longitude", value=-74.0, step=0.01)
        distance_from_last_txn = st.number_input("Distance from Last Transaction (km)", min_value=0.0, value=0.0, step=0.01)

    with col2:
        for i, col in enumerate(ohe.feature_names_in_):
            categorical_inputs[col] = st.selectbox(col, options=ohe.categories_[i])

    submitted = st.form_submit_button("🎯 Predict Fraud", type="primary")

# -------------------- Prediction --------------------
if submitted:
    trans_date_time = pd.to_datetime(f"{trans_date} {trans_time}")
    input_data = {
        'trans_date_trans_time': trans_date_time,
        'amt': amt,
        'time_diff': time_diff,
        'LoginAttempts': login_attempts,
        'txn_last_24h': txn_last_24h,
        'Account Age Days': account_age_days,
        'lat': lat,
        'long': long,
        'distance_from_last_txn_km': distance_from_last_txn
    }
    input_data.update(categorical_inputs)

    final_input = preprocess_input(input_data, ohe, scaler, feature_names)
    if final_input is not None:
        try:
            prediction = model.predict(final_input)

            st.subheader("🔍 Prediction Result")
            if prediction[0] == 1:
                st.markdown(
                    f"<div style='background-color:#ffe6e6;padding:15px;border-radius:8px;'>⚠️ "
                    f"**Fraudulent Transaction** detected with probability.</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background-color:#e6f4ea;padding:15px;border-radius:8px;'>✅ "
                    f"**Legitimate Transaction** with  fraud probability.</div>",
                    unsafe_allow_html=True)
            st.progress(probability / 100)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
