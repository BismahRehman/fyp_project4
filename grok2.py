import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing objects
rf_model = joblib.load('random_forest_fraud_detection_model.pkl')
ohe = joblib.load('ohe.pkl')
scaler = joblib.load('scaler (1).pkl')

# Define categorical and numerical columns
cat_columns = ['category', 'city', 'gender', 'state', 'job', 'Channel', 'IP Address']
numerical_cols = ['amt', 'LoginAttempts', 'trans_hour']


# Load dataset for dropdown options
@st.cache_data
def load_data():
    df1 = pd.read_csv('fraudTest.csv')
    df2 = pd.read_csv('bank_transactions_data_2.csv')
    md = pd.concat([df1, df2], axis=1)
    return md


md = load_data()


# Function to preprocess input data
def preprocess_input(data):
    input_df = pd.DataFrame([data])
    features_ohe = ohe.transform(input_df[cat_columns])
    encoded_cols = ohe.get_feature_names_out(cat_columns)
    features_ohe_df = pd.DataFrame(features_ohe, columns=encoded_cols, index=input_df.index)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    final_features = pd.concat([input_df[numerical_cols], features_ohe_df], axis=1)
    return final_features


# Streamlit app
st.title("Fraud Detection System")
st.write("Enter transaction details to predict if it's fraudulent.")

# Input form
with st.form("fraud_form"):
    st.header("Transaction Details")

    # Numerical inputs
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
    login_attempts = st.number_input("Login Attempts", min_value=0, value=1, step=1)
    trans_hour = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)

    # Categorical inputs
    category = st.selectbox("Category", options=md['category'].unique())
    city = st.selectbox("City", options=md['city'].unique())
    gender = st.selectbox("Gender", options=md['gender'].unique())
    state = st.selectbox("State", options=md['state'].unique())
    job = st.selectbox("Job", options=md['job'].unique())
    channel = st.selectbox("Channel", options=md['Channel'].unique())
    ip_address = st.text_input("IP Address", value=md['IP Address'].mode()[0])

    submitted = st.form_submit_button("Predict")

# Process prediction
if submitted:
    input_data = {
        'amt': amt,
        'LoginAttempts': login_attempts,
        'trans_hour': trans_hour,
        'category': category,
        'city': city,
        'gender': gender,
        'state': state,
        'job': job,
        'Channel': channel,
        'IP Address': ip_address
    }

    try:
        processed_input = preprocess_input(input_data)
        prediction = rf_model.predict(processed_input)
        probability = rf_model.predict_proba(processed_input)[0][1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"⚠️ This transaction is likely **Fraudulent** (Probability: {probability * 100:.2f}%)")
        else:
            st.success(
                f"✅ This transaction is likely **Non-Fraudulent** (Probability of Fraud: {probability * 100:.2f}%)")
    except ValueError as e:
        st.error(f"Error: {e}. Please ensure all inputs match training data categories.")

# Instructions
st.markdown("""
### How to Use
1. Enter transaction details in the form.
2. Click **Predict** to see if the transaction is fraudulent.
3. The result shows the prediction and fraud probability.
""")