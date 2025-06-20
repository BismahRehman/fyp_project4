import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# -------------------- Title --------------------
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("""
This app predicts whether a credit card transaction is fraudulent based on features from four datasets:  
**Fraudulent E-Commerce**, **Fraud Test**, **Credit Card Data**, and **Bank Transactions**.
""")

# -------------------- Load Model + Preprocessors --------------------
try:
    with open('model22.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler22.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ohe22.pkl', 'rb') as f:
        ohe = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Required files not found. Ensure `model22.pkl`, `scaler22.pkl`, and `ohe22.pkl` exist.")
    st.stop()

# -------------------- Column Names --------------------
categorical_cols = ['gender', 'city', 'state', 'Channel', 'Type of Card', 'Device Used', 'category', 'job']
numerical_cols = ['amt', 'LoginAttempts', 'trans_velocity', 'hour']

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv('fraudTest.csv')
    df2 = pd.read_csv('bank_transactions_data_2.csv')
    df3 = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data.csv')
    df4 = pd.read_csv('CreditCardData.csv')
    df_all = pd.concat([df1, df2, df3, df4], axis=0)
    df_all.columns = df_all.columns.str.strip()  # Remove any leading/trailing spaces
    return df_all

md = load_data()

# -------------------- Preprocessing --------------------
def preprocess_input(data, scaler, ohe):
    input_df = pd.DataFrame([data])
    input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'])
    input_df['hour'] = input_df['trans_date_trans_time'].dt.hour

    # Validate categorical inputs
    for i, col in enumerate(categorical_cols):
        if input_df[col].iloc[0] not in ohe.categories_[i]:
            st.error(f"Invalid value for {col}: '{input_df[col].iloc[0]}' is not recognized.")
            return None

    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    except ValueError as e:
        st.error(f"Scaling error: {str(e)}")
        return None

    try:
        input_ohe = ohe.transform(input_df[categorical_cols])
        encoded_cols = ohe.get_feature_names_out(categorical_cols)
        input_ohe_df = pd.DataFrame(input_ohe, columns=encoded_cols, index=input_df.index)
    except ValueError as e:
        st.error(f"Encoding error: {str(e)}")
        return None

    final_input = pd.concat([input_df[numerical_cols], input_ohe_df], axis=1)
    return final_input

# -------------------- Input Form --------------------
st.header("📝 Enter Transaction Details")

with st.form("transaction_form"):
    trans_date = st.date_input("Transaction Date", value=datetime.date.today())
    trans_time = st.time_input("Transaction Time", value=datetime.time(12, 0))
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
    login_attempts = st.number_input("Login Attempts", min_value=1, value=1, step=1)
    trans_velocity = st.number_input("Transaction Velocity (count/hour)", min_value=1.0, value=1.0, step=0.1)

    gender = st.selectbox("Gender", options=ohe.categories_[0])
    city = st.selectbox("City", options=ohe.categories_[1])
    state = st.selectbox("State", options=ohe.categories_[2])
    channel = st.selectbox("Channel", options=ohe.categories_[3])
    card_type = st.selectbox("Type of Card", options=ohe.categories_[4])
    device = st.selectbox("Device Used", options=ohe.categories_[5])
    category = st.selectbox("Transaction Category", options=ohe.categories_[6])
    job = st.selectbox("Job", options=ohe.categories_[7])

    submitted = st.form_submit_button("🎯 Predict Fraud")

# -------------------- Prediction --------------------
if submitted:
    trans_date_time = pd.to_datetime(f"{trans_date} {trans_time}")
    input_data = {
        'trans_date_trans_time': trans_date_time,
        'amt': amt,
        'LoginAttempts': login_attempts,
        'trans_velocity': trans_velocity,
        'gender': gender,
        'city': city,
        'state': state,
        'Channel': channel,
        'Type of Card': card_type,
        'Device Used': device,
        'category': category,
        'job': job
    }

    with st.spinner("Predicting..."):
        final_input = preprocess_input(input_data, scaler, ohe)
        if final_input is not None:
            try:
                prediction = model.predict(final_input)


                st.subheader("🔍 Prediction Result")
                if prediction[0] == 1:
                    st.error(f"⚠️ Fraudulent Transaction risked. be carefull.")
                else:
                    st.success(f"✅ Non-Fraudulent Transaction ")

                st.progress(probability / 100)
                st.markdown(f"`Fraud Probability: {probability:.2f}%`")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.info("Please check inputs and try again.")

# -------------------- Charts --------------------
st.header("📊 Data Insights")
st.markdown("Insights from the combined transaction dataset.")

# Dynamic fraud count if column exists
if 'is_fraud' in md.columns:
    counts = md['is_fraud'].value_counts().sort_index()
    fraud_counts = [counts.get(0, 0), counts.get(1, 0)]
else:
    fraud_counts = [95, 5]  # Fallback

labels = ['Non-Fraud', 'Fraud']
colors = ['#4CAF50', '#FF5733']
st.subheader("Fraud vs. Non-Fraud Transactions")
fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
st.pyplot(fig)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(f"Built with ❤️ using Streamlit | Model: Random Forest | © {datetime.datetime.now().year} Bismah")