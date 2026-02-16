import streamlit as st
import numpy as np
import pickle

# Load saved files
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
le_dict = pickle.load(open("model/label_encoder.pkl", "rb"))

st.title("Credit Card Fraud Detection System")

# ===============================
# Numeric Inputs
# ===============================

transaction_amount = st.number_input("Transaction Amount")
transaction_hour = st.number_input("Transaction Hour (0-23)")
customer_age = st.number_input("Customer Age")
previous_fraud_count = st.number_input("Previous Fraud Count")
avg_transaction_amount = st.number_input("Average Transaction Amount")
account_age_days = st.number_input("Account Age (Days)")
num_transactions_24h = st.number_input("Transactions (Last 24h)")
num_transactions_7d = st.number_input("Transactions (Last 7 Days)")
risk_score = st.number_input("Risk Score")

# ===============================
# Categorical Inputs
# ===============================

merchant_category = st.selectbox("Merchant Category", le_dict["merchant_category"].classes_)
customer_location = st.selectbox("Customer Location", le_dict["customer_location"].classes_)
device_type = st.selectbox("Device Type", le_dict["device_type"].classes_)
card_type = st.selectbox("Card Type", le_dict["card_type"].classes_)
transaction_type = st.selectbox("Transaction Type", le_dict["transaction_type"].classes_)
is_international = st.selectbox("International Transaction?", [0,1])
is_weekend = st.selectbox("Weekend?", [0,1])

# ===============================
# Prediction
# ===============================

if st.button("Predict"):

    # Encode categorical values
    merchant_category = le_dict["merchant_category"].transform([merchant_category])[0]
    customer_location = le_dict["customer_location"].transform([customer_location])[0]
    device_type = le_dict["device_type"].transform([device_type])[0]
    card_type = le_dict["card_type"].transform([card_type])[0]
    transaction_type = le_dict["transaction_type"].transform([transaction_type])[0]

    input_data = np.array([[ 
        transaction_amount,
        transaction_hour,
        merchant_category,
        customer_age,
        customer_location,
        device_type,
        card_type,
        transaction_type,
        previous_fraud_count,
        avg_transaction_amount,
        account_age_days,
        num_transactions_24h,
        num_transactions_7d,
        is_international,
        is_weekend,
        risk_score
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
