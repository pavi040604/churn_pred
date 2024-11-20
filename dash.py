import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(r"C:\Users\tnv10\xgboost_model.pkl")

# Define the expected features
expected_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This application predicts whether a customer is likely to churn based on their details.")

# Input fields
st.header("Enter Customer Details:")
senior_citizen = st.selectbox("Is the customer a senior citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
tenure = st.number_input("Tenure (in months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges (in USD)", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges (in USD)", min_value=0.0, value=600.0)

# Predict button
if st.button("Predict Churn"):
    # Prepare input data
    input_data = pd.DataFrame({
        'SeniorCitizen': [senior_citizen],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
    })

    # Ensure column order matches model training
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Make prediction
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]

        if prediction[0] == 1:
            st.error(f"The customer is likely to churn. (Probability: {probability:.2f})")
        else:
            st.success(f"The customer is not likely to churn. (Probability: {1 - probability:.2f})")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
