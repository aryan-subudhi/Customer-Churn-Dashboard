import streamlit as st
import pandas as pd
import sqlite3
import joblib

# Page Title
st.title("Customer Churn Dashboard")

# Load CSV data
df = pd.read_csv("data/churn.csv")

df.columns = df.columns.str.strip()

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Churn Distribution
st.subheader("Churn Distribution")
st.bar_chart(df["Churn Label"].value_counts())

# Connect to SQL database
conn = sqlite3.connect("customer_churn.db")

# Read SQL table
sql_data = pd.read_sql("SELECT * FROM customers LIMIT 5", conn)

st.subheader("SQL Database Preview")
st.write(sql_data)

conn.close()

# Load trained model
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.subheader("Predict Customer Churn")

# User Inputs
tenure = st.slider("Tenure Months", 1, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

# Create input dataframe
input_data = pd.DataFrame({
    "Tenure Months": [tenure],
    "Monthly Charges": [monthly_charges]
})

# Add missing columns
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns correctly
input_data = input_data[model_columns]

# Predict button
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:

        st.error(
            f"⚠️ Customer likely to churn! Risk Score: {probability:.2f}"
        )

        st.write("Suggested Retention Action:")
        st.write("- Offer discount")
        st.write("- Improve engagement")
        st.write("- Provide loyalty benefits")

    else:

        st.success(
            f"✅ Customer likely to stay. Risk Score: {probability:.2f}"
        )