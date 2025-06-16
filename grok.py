import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Load the saved RandomForest model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the original dataset to get unique values for categorical columns
try:
    md = pd.read_csv('fraudTest.csv')
except FileNotFoundError:
    st.error("Dataset file 'fraudTest.csv' not found. Please ensure it's in the same directory as this script.")
    st.stop()

# Define the categorical and numerical features
categorical_columns = ['category', 'gender', 'city', 'state', 'job']
numerical_columns = ['amt']

# Get unique values from the dataset
categories = md['category'].unique().tolist()
genders = md['gender'].unique().tolist()
cities = md['city'].unique().tolist()
states = md['state'].unique().tolist()
jobs = md['job'].unique().tolist()

# Print arrays and their lengths for debugging
st.subheader("Debug: Categorical Arrays and Their Lengths")
st.write("Categories:", categories)
st.write(f"Length of Categories: {len(categories)}")
st.write("Genders:", genders)
st.write(f"Length of Genders: {len(genders)}")
st.write("Cities:", cities)
st.write(f"Length of Cities: {len(cities)}")
st.write("States:", states)
st.write(f"Length of States: {len(states)}")
st.write("Jobs:", jobs)
st.write(f"Length of Jobs: {len(jobs)}")

# Pad arrays to the maximum length to avoid length mismatch
max_length = max(len(categories), len(genders), len(cities), len(states), len(jobs))
categories = categories + [categories[0]] * (max_length - len(categories))
genders = genders + [genders[0]] * (max_length - len(genders))
cities = cities + [cities[0]] * (max_length - len(cities))
states = states + [states[0]] * (max_length - len(states))
jobs = jobs + [jobs[0]] * (max_length - len(jobs))

# Verify lengths after padding
st.write("\nAfter Padding:")
st.write(f"Length of Categories: {len(categories)}")
st.write(f"Length of Genders: {len(genders)}")
st.write(f"Length of Cities: {len(cities)}")
st.write(f"Length of States: {len(states)}")
st.write(f"Length of Jobs: {len(jobs)}")

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Create a DataFrame for fitting the encoder
encoder_data = pd.DataFrame({
    'category': categories,
    'gender': genders,
    'city': cities,
    'state': states,
    'job': jobs
})

# Fit the encoder
encoder.fit(encoder_data[categorical_columns])

# Streamlit app
st.title("Fraud Detection System")
st.write("Enter transaction details to predict if it's fraudulent.")

# Create input fields
with st.form("prediction_form"):
    st.subheader("Transaction Details")

    # Numerical input
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.0, step=0.01)

    # Categorical inputs
    category = st.selectbox("Category", options=categories)
    gender = st.selectbox("Gender", options=genders)
    city = st.selectbox("City", options=cities)
    state = st.selectbox("State", options=states)
    job = st.selectbox("Job", options=jobs)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Process input and make prediction
if submitted:
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'amt': [amt],
        'category': [category],
        'gender': [gender],
        'city': [city],
        'state': [state],
        'job': [job]
    })

    # Encode categorical variables
    encoded_values = encoder.transform(input_data[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_values, columns=encoded_columns)

    # Combine numerical and encoded categorical features
    input_features = pd.concat([input_data[numerical_columns], encoded_df], axis=1)

    # Ensure the feature order matches the training data
    model_features = model.feature_names_in_
    input_features = input_features.reindex(columns=model_features, fill_value=0)

    # Make prediction
    prediction = model.predict(input_features)
    prediction_prob = model.predict_proba(input_features)[0]

    # Display results
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("**Fraudulent Transaction Detected!**")
    else:
        st.success("**Non-Fraudulent Transaction**")

    st.write(f"Probability of Fraud: {prediction_prob[1]:.2%}")
    st.write(f"Probability of Non-Fraud: {prediction_prob[0]:.2%}")

# Footer
st.markdown("---")
st.write("Built with Streamlit and scikit-learn. Model: RandomForestClassifier")