import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* General styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    h1 {
        color: #1a3c34;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    h2 {
        color: #1a3c34;
        font-weight: 600;
        margin-top: 20px;
    }
    .stMarkdown p {
        color: #4a4a4a;
        font-size: 16px;
    }
    /* Input form styling */
    .stForm {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #e55a4f;
    }
    /* Prediction result styling */
    .prediction-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        text-align: center;
    }
    .fraud {
        background-color: #ffe6e6;
        border: 2px solid #ff6f61;
    }
    .non-fraud {
        background-color: #e6f4ea;
        border: 2px solid #1a3c34;
    }
    .stProgress .st-bo {
        background-color: #ff6f61;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #4a4a4a;
        font-size: 14px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.title("🛡️ Credit Card Fraud Detection")
st.markdown("""
Predict whether a credit card transaction is fraudulent using a Random Forest model trained on multiple datasets.  
Enter transaction details below to get started.
""")

# -------------------- Load Model + Preprocessors --------------------
try:
    model = joblib.load('random_forest_model99.pkl')
    ohe = joblib.load('one_hot_encoder99.pkl')
    scaler = joblib.load('standard_scaler99.pkl')
    feature_names = joblib.load('feature_names99.pkl')
except FileNotFoundError:
    st.error("❌ Required files not found. Ensure `random_forest_model99.pkl`, `one_hot_encoder99.pkl`, `standard_scaler99.pkl`, and `feature_names99.pkl` exist.")
    st.stop()

# -------------------- Column Names --------------------
categorical_cols = ['gender', 'city', 'state', 'Type of Transaction', 'Type of Card', 'Device Used', 'category', 'job']
numerical_cols = ['amt', 'hour', 'day', 'day_of_week', 'month', 'time_diff', 'is_night', 'LoginAttempts', 'txn_last_24h']

# -------------------- Load Dataset for Visualizations --------------------
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv('Fraudulent_E-Commerce_Transaction_Data.csv')
        df2 = pd.read_csv('fraudTest.csv')
        df3 = pd.read_csv('CreditCardData.csv')
        df4 = pd.read_csv('bank_transactions_data_2.csv')
        # Preprocess df3 to match training
        df3['Amount'] = df3['Amount'].str.replace('£', '', regex=False).astype(float)
        df3.rename(columns={'Amount': 'amt', 'Merchant Group': 'category', 'Gender': 'gender', 'Fraud': 'is_fraud'}, inplace=True)
        # Preprocess df1
        df1['Device Used'] = df1['Device Used'].replace({'tablet': 'laptop', 'desktopp': 'unknown'})
        # Concatenate
        df = pd.concat([df1, df2, df3, df4], axis=0)
        df.columns = df.columns.str.strip()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
        # Handle missing values
        df['amt'] = df['amt'].fillna(df['amt'].median())
        df['LoginAttempts'] = df['LoginAttempts'].fillna(df['LoginAttempts'].median())
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        df = df[df['is_fraud'].notnull()].copy().reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset files not found. Ensure `Fraudulent_E-Commerce_Transaction_Data.csv`, `fraudTest.csv`, `CreditCardData.csv`, and `bank_transactions_data_2.csv` exist.")
        st.stop()

df = load_data()

# -------------------- Preprocessing --------------------
def preprocess_input(data, ohe, scaler, feature_names):
    input_df = pd.DataFrame([data])
    input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'], errors='coerce')
    input_df['hour'] = input_df['trans_date_trans_time'].dt.hour
    input_df['day_of_week'] = input_df['trans_date_trans_time'].dt.dayofweek
    input_df['day'] = input_df['trans_date_trans_time'].dt.day
    input_df['month'] = input_df['trans_date_trans_time'].dt.month
    input_df['is_night'] = input_df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    # Validate categorical inputs
    for i, col in enumerate(categorical_cols):
        if input_df[col].iloc[0] not in ohe.categories_[i]:
            st.error(f"Invalid value for {col}: '{input_df[col].iloc[0]}' is not recognized.")
            return None

    try:
        # Scale numerical features
        input_scaled = scaler.transform(input_df[numerical_cols])
        input_scaled_df = pd.DataFrame(input_scaled, columns=numerical_cols, index=input_df.index)
        # One-hot encode categorical columns
        input_ohe = ohe.transform(input_df[categorical_cols])
        encoded_cols = ohe.get_feature_names_out(categorical_cols)
        input_ohe_df = pd.DataFrame(input_ohe, columns=encoded_cols, index=input_df.index)
        # Combine scaled numerical and encoded features
        final_input = pd.concat([input_scaled_df, input_ohe_df], axis=1)
        # Ensure all expected features are present
        final_input = final_input.reindex(columns=feature_names, fill_value=0)
        return final_input
    except ValueError as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

# -------------------- Input Form --------------------
st.header("📝 Enter Transaction Details")
with st.container():
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            trans_date = st.date_input("Transaction Date", value=datetime.date.today())
            trans_time = st.time_input("Transaction Time", value=datetime.time(12, 0))
            amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
            login_attempts = st.number_input("Login Attempts", min_value=0, value=1, step=1)
            time_diff = st.number_input("Time Since Last Transaction (seconds)", min_value=0.0, value=0.0, step=0.1)
            txn_last_24h = st.number_input("Transactions in Last 24 Hours", min_value=0, value=1, step=1)
        with col2:
            gender = st.selectbox("Gender", options=ohe.categories_[0])
            city = st.selectbox("City", options=ohe.categories_[1])
            state = st.selectbox("State", options=ohe.categories_[2])
            type_of_transaction = st.selectbox("Type of Transaction", options=ohe.categories_[3])
            type_of_card = st.selectbox("Type of Card", options=ohe.categories_[4])
            device_used = st.selectbox("Device Used", options=ohe.categories_[5])
            category = st.selectbox("Transaction Category", options=ohe.categories_[6])
            job = st.selectbox("Job", options=ohe.categories_[7])

        submitted = st.form_submit_button("🎯 Predict Fraud", type="primary")

# -------------------- Prediction --------------------
if submitted:
    trans_date_time = pd.to_datetime(f"{trans_date} {trans_time}")
    input_data = {
        'trans_date_trans_time': trans_date_time,
        'amt': amt,
        'hour': 0,  # Will be overwritten by preprocessing
        'day': 0,   # Will be overwritten
        'day_of_week': 0,  # Will be overwritten
        'month': 0,  # Will be overwritten
        'is_night': 0,  # Will be overwritten
        'time_diff': time_diff,
        'LoginAttempts': login_attempts,
        'txn_last_24h': txn_last_24h,
        'gender': gender,
        'city': city,
        'state': state,
        'Type of Transaction': type_of_transaction,
        'Type of Card': type_of_card,
        'Device Used': device_used,
        'category': category,
        'job': job
    }

    with st.spinner("Predicting..."):
        final_input = preprocess_input(input_data, ohe, scaler, feature_names)
        if final_input is not None:
            try:
                prediction = model.predict(final_input)
                probability = model.predict_proba(final_input)[0][1] * 100

                st.subheader("🔍 Prediction Result")
                result_class = "fraud" if prediction[0] == 1 else "non-fraud"
                result_text = f"⚠️ Fraudulent Transaction with {probability:.2f}% probability." if prediction[0] == 1 else f"✅ Non-Fraudulent Transaction with {100 - probability:.2f}% probability."
                st.markdown(f'<div class="prediction-box {result_class}">{result_text}</div>', unsafe_allow_html=True)
                st.progress(probability / 100)
                st.markdown(f"<p style='text-align: center; color: #4a4a4a;'>Fraud Probability: {probability:.2f}%</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.info("Please check inputs and try again.")

# -------------------- Charts --------------------
st.header("📊 Data Insights")
st.markdown("Explore trends in the transaction dataset.")

# Fraud vs. Non-Fraud Pie Chart
if 'is_fraud' in df.columns:
    counts = df['is_fraud'].value_counts().sort_index()
    fraud_counts = [counts.get(0, 0), counts.get(1, 0)]
else:
    fraud_counts = [95, 5]  # Fallback

st.subheader("Fraud vs. Non-Fraud Transactions")

# Replace __FRAUD_COUNTS__ in the chart
fraud_counts_str = f"[{fraud_counts[0]}, {fraud_counts[1]}]"
st.markdown(f"<script>window.chartData = {{...window.chartData, 'fraud_pie': {{...window.chartData['fraud_pie'], data: {{...window.chartData['fraud_pie'].data, datasets: [{{...window.chartData['fraud_pie'].data.datasets[0], data: {fraud_counts_str}}}]}}}}</script>", unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown(f'<div class="footer">Built with ❤️ using Streamlit | Model: Random Forest | © {datetime.datetime.now().year} Bismah</div>', unsafe_allow_html=True)