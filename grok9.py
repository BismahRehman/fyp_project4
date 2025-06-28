import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from streamlit_option_menu import option_menu

# -------------------- Set Background --------------------
def set_background(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }}
    h1, h2 {{
        color: #1a3c34;
        text-align: center;
    }}
    .stButton>button {{
        background-color: #ff6f61;
        color: white;
        border-radius: 8px;
        width: 100%;
    }}
    .stButton>button:hover {{
        background-color: #e55a4f;
    }}
    .prediction-box {{
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        text-align: center;
    }}
    .fraud {{
        background-color: #ffe6e6;
        border: 2px solid #ff6f61;
    }}
    .non-fraud {{
        background-color: #e6f4ea;
        border: 2px solid #1a3c34;
    }}
    .footer {{
        text-align: center;
        color: #4a4a4a;
        font-size: 14px;
        margin-top: 30px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------- Page Config --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
set_background("image.jpeg")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("ch_prev_ui.png", use_container_width=True)
    page = option_menu("Main Menu", ["Home", "Analytics", "About", "Contact"],
                       icons=['house', 'bar-chart-line', 'info-circle', 'envelope'],
                       menu_icon="cast", default_index=0,
                       styles={"nav-link-selected": {"background-color": "green"}})

# -------------------- Load Model --------------------
try:
    model = joblib.load('random_forest_model55.pkl')
    ohe = joblib.load('one_hot_encoder55.pkl')
    feature_names = joblib.load('feature_names55.pkl')
except FileNotFoundError:
    st.error("❌ Required files not found.")
    st.stop()

# -------------------- Column Names --------------------
categorical_cols = ['gender', 'city', 'state', 'Type of Card', 'Day of Week', 'Type of Transaction', 'category', 'job']
numerical_cols = ['amt', 'Time']

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv('fraudTest.csv')
    df2 = pd.read_csv('CreditCardData.csv')
    if df2['Amount'].dtype == 'object':
        df2['Amount'] = df2['Amount'].str.replace('£', '', regex=False).astype(float)
    df2.rename(columns={'Amount': 'amt', 'Merchant Group': 'category', 'Gender': 'gender', 'Fraud': 'is_fraud'}, inplace=True)
    df = pd.concat([df1, df2], axis=0)
    df.columns = df.columns.str.strip()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['Time'] = df['trans_date_trans_time'].dt.hour
    df['Day of Week'] = df['trans_date_trans_time'].dt.day_name()
    return df

df = load_data()

# -------------------- Preprocessing --------------------
def preprocess_input(data, ohe, feature_names):
    input_df = pd.DataFrame([data])
    input_df['trans_date_trans_time'] = pd.to_datetime(input_df['trans_date_trans_time'], errors='coerce')
    input_df['Time'] = input_df['trans_date_trans_time'].dt.hour
    input_df['Day of Week'] = input_df['trans_date_trans_time'].dt.day_name()
    for i, col in enumerate(categorical_cols):
        if input_df[col].iloc[0] not in ohe.categories_[i]:
            st.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
            return None
    input_ohe = ohe.transform(input_df[categorical_cols])
    encoded_cols = ohe.get_feature_names_out(categorical_cols)
    input_ohe_df = pd.DataFrame(input_ohe, columns=encoded_cols, index=input_df.index)
    final_input = pd.concat([input_df[numerical_cols], input_ohe_df], axis=1)
    final_input = final_input.reindex(columns=feature_names, fill_value=0)
    return final_input

# -------------------- Pages --------------------
if page == "Home":
    st.title("🛡️ Credit Card Fraud Detection")
    st.markdown("Predict whether a transaction is fraudulent using a Random Forest model.")

    st.header("📝 Enter Transaction Details")
    with st.form("transaction_form"):
        trans_date = st.date_input("Transaction Date", value=None)
        trans_time = st.time_input("Transaction Time", value=None)
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0, step=0.01)
        gender = st.selectbox("Gender", options=["Select"] + sorted(df['gender'].dropna().unique()))
        city = st.selectbox("City", options=["Select"] + sorted(df['city'].dropna().unique()))
        state = st.selectbox("State", options=["Select"] + sorted(df['state'].dropna().unique()))
        type_of_card = st.selectbox("Type of Card", options=["Select"] + sorted(df['Type of Card'].dropna().unique()))
        type_of_transaction = st.selectbox("Type of Transaction",
                                           options=["Select"] + sorted(df['Type of Transaction'].dropna().unique()))
        category = st.selectbox("Transaction Category", options=["Select"] + sorted(df['category'].dropna().unique()))
        job = st.selectbox("Job", options=["Select"] + sorted(df['job'].dropna().unique()))
        submitted = st.form_submit_button("🎯 Predict Fraud")

    if submitted:
        if not trans_date or not trans_time:
            st.warning("⚠️ Please select date and time.")
        else:
            trans_date_time = pd.to_datetime(f"{trans_date} {trans_time}")
            input_data = {
                'trans_date_trans_time': trans_date_time,
                'amt': amt,
                'Time': trans_date_time.hour,
                'gender': gender,
                'city': city,
                'state': state,
                'Type of Card': type_of_card,
                'Day of Week': trans_date_time.day_name(),
                'Type of Transaction': type_of_transaction,
                'category': category,
                'job': job
            }
            with st.spinner("Predicting..."):
                final_input = preprocess_input(input_data, ohe, feature_names)
                if final_input is not None:
                    prediction = model.predict(final_input)
                    probability = model.predict_proba(final_input)[0][1] * 100
                    result_class = "fraud" if prediction[0] == 1 else "non-fraud"
                    result_text = "⚠️ Fraudulent Transaction" if prediction[0] == 1 else "✅ Non-Fraudulent Transaction"
                    st.markdown(f'<div class="prediction-box {result_class}">{result_text}</div>', unsafe_allow_html=True)
                    st.progress(probability / 100)
                    st.markdown(f"<p style='text-align: center;'>Fraud Probability: {probability:.2f}%</p>", unsafe_allow_html=True)

elif page == "Analytics":
    st.title("📊 Data Insights")
    selected_analysis = option_menu(
        menu_title=None,
        options=['fraud transaction', 'gender analysis', 'category analysis', 'Amount by Fraud','location graph'],
        icons=['house', 'person', 'tags', 'credit-card'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0px", "background-color": "#f0f0f0", "margin-bottom": "20px"},
            "nav-link": {"font-size": "16px", "color": "black", "min-width": "fit-content", "text-align": "center", "font-weight": "normal"},
            "nav-link-selected": {"background-color": "#4a90e2", "color": "white", "font-weight": "normal"},
        }
    )

    if selected_analysis == "fraud transaction":
        fraud_counts = df['is_fraud'].value_counts()
        labels = ['Non-Fraud', 'Fraud']
        colors = ['#4CAF50', '#FF5733']
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(fraud_counts, labels=labels, autopct='%1.0f%%', colors=colors, startangle=140)
        ax.set_title('Fraud vs. Non-Fraud Transactions')
        st.pyplot(fig)

    elif selected_analysis == "gender analysis":
        # Countplot by Gender
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df.dropna(subset=['gender', 'is_fraud']), x='gender', hue='is_fraud', palette='coolwarm',
                      ax=ax)
        st.pyplot(fig)

    elif selected_analysis == "category analysis":
        # Countplot by Category
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df.dropna(subset=['category', 'is_fraud']), x='category', hue='is_fraud', palette='tab10',
                      ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig3)


    elif selected_analysis == "Amount by Fraud":
        fig, ax = plt.subplots(figsize=(8, 5))
        df.groupby('is_fraud')['amt'].mean().plot(kind='bar', color=['#FFB6C1', '#9370DB'], ax=ax)
        ax.set_title('Avg Amount by Fraud Status')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Fraud', 'Fraud'])
        st.pyplot(fig)

    elif selected_analysis == "location graph":
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter_df = df.dropna(subset=['lat', 'long', 'is_fraud'])
        sns.scatterplot(x=scatter_df['long'], y=scatter_df['lat'], hue=scatter_df['is_fraud'], palette=['blue', 'red'],
                        ax=ax)
        ax.set_title('Transaction Locations (Red = Fraud)')
        st.pyplot(fig)

elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    Built by **Bismah**, a tech enthusiast from Pakistan 🇵🇰  
    **Stack**: Streamlit, Random Forest, Pandas, Seaborn  
    **Project**: Final Year Fraud Detection App 💻  
    """)

elif page == "Contact":
    st.title("📨 Contact")
    st.markdown("""
    Want to connect or collaborate?

    - 📧 Email: bismah@example.com  
    - 👩‍💻 GitHub: [your-github](https://github.com/your-profile)  
    - 💼 LinkedIn: [your-linkedin](https://linkedin.com/in/your-profile)
    """)

# -------------------- Footer --------------------
st.markdown(f'<div class="footer">Made with ❤️ by Bismah | © {datetime.datetime.now().year}</div>', unsafe_allow_html=True)