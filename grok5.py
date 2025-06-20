import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model, encoder, and feature names
model = joblib.load('random_forest_model55.pkl')
ohe = joblib.load('one_hot_encoder55.pkl')
feature_names = joblib.load('feature_names55.pkl')

# Load dataset for visualizations
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv('fraudTest.csv')
        df2 = pd.read_csv('CreditCardData.csv')
        # Preprocess df2 to match training
        df2['Amount'] = df2['Amount'].str.replace('£', '', regex=False).astype(float)
        df2.rename(columns={'Amount': 'amt', 'Merchant Group': 'category', 'Gender': 'gender', 'Fraud': 'is_fraud'}, inplace=True)
        df = pd.concat([df1, df2], axis=0, ignore_index=True)
        df.columns = df.columns.str.strip()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['Time'] = df['trans_date_trans_time'].dt.hour
        df['Day of Week'] = df['trans_date_trans_time'].dt.day_name()
        return df
    except FileNotFoundError:
        st.error("❌ Dataset files not found. Ensure `fraudTest.csv` and `CreditCardData.csv` are in the correct directory.")
        st.stop()

df = load_data()

# Streamlit app title
st.title("Credit Card Fraud Detection")

# Sidebar for visualizations
st.sidebar.header("Visualizations")
viz_option = st.sidebar.selectbox(
    "Select Visualization",
    ["None", "Fraud vs. Non-Fraud Pie Chart", "Average Transaction Amount by Fraud Status", "Fraud by Category"]
)

# Display visualizations based on selection
if viz_option == "Fraud vs. Non-Fraud Pie Chart":
    st.subheader("Fraud vs. Non-Fraud Transactions")
    fraud_counts = df['is_fraud'].value_counts()
    labels = ['Non-Fraud', 'Fraud']
    colors = ['#4CAF50', '#FF5733']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'fontsize': 12})
    ax.set_title('Fraud vs. Non-Fraud Transactions', fontsize=14)
    st.pyplot(fig)

elif viz_option == "Average Transaction Amount by Fraud Status":
    st.subheader("Average Transaction Amount by Fraud Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby('is_fraud')['amt'].mean().plot(kind='bar', color=['#FFB6C1', '#9370DB'], ax=ax)
    ax.set_title('Average Transaction Amount by Fraud Status', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
    ax.set_ylabel('Average Transaction Amount', fontsize=12)
    ax.set_xlabel('Fraud Status', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#DDA0DD')
    st.pyplot(fig)

elif viz_option == "Fraud by Category":
    st.subheader("Fraud vs. Non-Fraud Transactions by Category")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=df, x='category', hue='is_fraud', palette='tab10', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Fraud vs. Non-Fraud Transactions by Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transaction Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(title='Fraud Status', labels=['Non-Fraud', 'Fraud'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Create input fields for the features
st.header("Enter Transaction Details")

# Numerical inputs
amt = st.number_input("Transaction Amount (£)", min_value=0.0, value=0.0, step=0.01)
time = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)

# Categorical inputs using ohe.categories_
categorical_cols = ['gender', 'city', 'state', 'Type of Card', 'Day of Week', 'Type of Transaction', 'category', 'job']
input_data = {}

# Map categorical columns to their options from the encoder
for i, col in enumerate(categorical_cols):
    if col == 'Day of Week':
        # Hardcode day of week options to ensure consistency
        options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    else:
        options = list(ohe.categories_[i])
    input_data[col] = st.selectbox(f"{col}", options=options, key=col)

# Prepare input data as a DataFrame
input_data['amt'] = amt
input_data['Time'] = time
input_df = pd.DataFrame([input_data])

# Preprocessing function
def preprocess_input(data, ohe, feature_names):
    try:
        # One-hot encode categorical columns
        input_ohe = ohe.transform(data[categorical_cols])
        encoded_cols = ohe.get_feature_names_out(categorical_cols)
        input_ohe_df = pd.DataFrame(input_ohe, columns=encoded_cols, index=data.index)
        # Combine numerical and encoded features
        numerical_cols = ['amt', 'Time']
        final_input = pd.concat([data[numerical_cols], input_ohe_df], axis=1)
        # Ensure all expected features are present
        final_input = final_input.reindex(columns=feature_names, fill_value=0)
        return final_input
    except ValueError as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

# Button to make prediction
if st.button("Predict Fraud"):
    final_input = preprocess_input(input_df, ohe, feature_names)
    if final_input is not None:
        # Make prediction
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)[0][1]  # Probability of fraud
        # Display result
        if prediction[0] == 1:
            st.error(f"⚠️ This transaction is predicted to be **Fraudulent** ")
        else:
            st.success(f"✅ This transaction is predicted to be **Non-Fraudulent** ")

# Add instructions
st.markdown("""
### Instructions
1. Select or enter the transaction details in the fields above.
2. Choose a visualization from the sidebar to explore the dataset.
3. Click the "Predict Fraud" button to see the prediction.
4. The model will indicate whether the transaction is likely fraudulent or not, along with the probability of fraud.
""")