import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the saved model, encoder, scaler, and feature names
model = joblib.load('random_forest_model77 (1).pkl')
ohe = joblib.load('one_hot_encoder77 (1).pkl')
scaler = joblib.load('standard_scaler77.pkl')
feature_names = joblib.load('feature_names77 (1).pkl')


# Load dataset for visualizations
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('CreditCardData.csv')
        df['Amount'] = df['Amount'].str.replace('[£$,]', '', regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Time'] = df['Time'].astype(int)
        df['Day of Week'] = df['Date'].dt.day_name()
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Ensure `CreditCardData.csv` is in the correct directory.")
        st.stop()


df = load_data()

# Streamlit app title
st.title("Credit Card Fraud Detection")

# Sidebar for visualizations
st.sidebar.header("Visualizations")
viz_option = st.sidebar.selectbox(
    "Select Visualization",
    ["None", "Fraud vs. Non-Fraud Pie Chart", "Average Transaction Amount by Fraud Status",
     "Fraud by Category", "Fraud by Country", "Transaction Amount Distribution"]
)

# Display visualizations based on selection
if viz_option == "Fraud vs. Non-Fraud Pie Chart":
    st.subheader("Fraud vs. Non-Fraud Transactions")
    fraud_counts = df['Fraud'].value_counts()
    labels = ['Non-Fraud', 'Fraud']
    colors = ['#4CAF50', '#FF5733']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'fontsize': 12})
    ax.set_title('Fraud vs. Non-Fraud Transactions', fontsize=14)
    st.pyplot(fig)

elif viz_option == "Average Transaction Amount by Fraud Status":
    st.subheader("Average Transaction Amount by Fraud Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby('Fraud')['Amount'].mean().plot(kind='bar', color=['#FFB6C1', '#9370DB'], ax=ax)
    ax.set_title('Average Transaction Amount by Fraud Status', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
    ax.set_ylabel('Average Transaction Amount', fontsize=12)
    ax.set_xlabel('Fraud Status', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#DDA0DD')
    st.pyplot(fig)

elif viz_option == "Fraud by Category":
    st.subheader("Fraud vs. Non-Fraud Transactions by Category")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x='Merchant Group', hue='Fraud', palette='tab10', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Fraud vs. Non-Fraud Transactions by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(title='Fraud Status', labels=['Non-Fraud', 'Fraud'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif viz_option == "Fraud by Country":
    st.subheader("Fraud vs. Non-Fraud Transactions by Country")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x='Country of Residence', hue='Fraud', palette=['blue', 'red'], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title('Fraud vs Non-Fraud Transactions by Country', fontsize=14, fontweight='bold')
    ax.set_xlabel('Country of Residence', fontsize=12)
    ax.set_ylabel('Transaction Count', fontsize=12)
    ax.legend(title='Fraud')
    st.pyplot(fig)

elif viz_option == "Transaction Amount Distribution":
    st.subheader("Transaction Amount Distribution")
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-5000', '5000+']
    df['amount_range'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    df['amount_range'].value_counts().sort_index().plot.bar(color='#7EB77F', edgecolor='black', ax=ax)
    ax.set_title('Transaction Amount Distribution (Custom Bins)', fontsize=14)
    ax.set_xlabel('Amount Range', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis='y', linestyle='--')
    st.pyplot(fig)

# Create input fields for the features
st.header("Enter Transaction Details")

# Numerical inputs
amount = st.number_input("Transaction Amount (£)", min_value=0.0, value=0.0, step=0.01)

# Date input
date = st.date_input("Transaction Date", value=datetime.today())

# Transaction hour dropdown
time_options = list(range(24))
time = st.selectbox("Transaction Hour (0-23)", options=time_options, index=12)

# Categorical inputs using ohe.categories_
categorical_cols = ['Gender', 'Country of Residence', 'Country of Transaction', 'Type of Card','Day of Week',
                    'Type of Transaction', 'Merchant Group']
input_data = {}

# Map categorical columns to their options from the encoder
for i, col in enumerate(categorical_cols):
    options = list(ohe.categories_[i])
    input_data[col] = st.selectbox(f"{col}", options=options, key=col)

# Derive Day of Week from Date
day_of_week = date.strftime('%A')
input_data['Day of Week'] = day_of_week

# Prepare input data as a DataFrame
input_data['Amount'] = amount
input_data['Time'] = time
input_df = pd.DataFrame([input_data])


# Preprocessing function
def preprocess_input(data, ohe, scaler, feature_names):
    try:
        # Scale numerical features
        numerical_cols = ['Amount', 'Time']
        scaled_data = scaler.transform(data[numerical_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=data.index)

        # One-hot encode categorical columns
        categorical_cols_ohe = ['Gender', 'Country of Residence', 'Country of Transaction',
                                'Type of Card', 'Day of Week', 'Type of Transaction', 'Merchant Group']
        categorical_data = ohe.transform(data[categorical_cols_ohe])
        encoded_cols = ohe.get_feature_names_out(categorical_cols_ohe)
        categorical_df = pd.DataFrame(categorical_data, columns=encoded_cols, index=data.index)

        # Combine scaled numerical and encoded categorical features
        final_input = pd.concat([scaled_df, categorical_df], axis=1)

        # Ensure all expected features are present
        final_input = final_input.reindex(columns=feature_names, fill_value=0)
        return final_input
    except ValueError as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None


# Button to make prediction
if st.button("Predict Fraud"):
    final_input = preprocess_input(input_df, ohe, scaler, feature_names)
    if final_input is not None:
        # Make prediction
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)[0][1]  # Probability of fraud
        # Display result
        if prediction[0] == 1:
            st.error(f"⚠️ This transaction is predicted to be **Fraudulent** (Probability: {probability:.2%})")
        else:
            st.success(
                f"✅ This transaction is predicted to be **Non-Fraudulent** (Probability of Fraud: {probability:.2%})")

# Add instructions
st.markdown("""
### Instructions
1. Enter the transaction details, including the date and hour, in the fields above.
2. Select a visualization from the sidebar to explore the dataset.
3. Click the "Predict Fraud" button to see the prediction.
4. The model will indicate whether the transaction is likely fraudulent or not, along with the probability of fraud.
""")