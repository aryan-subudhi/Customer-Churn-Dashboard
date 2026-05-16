import streamlit as st
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("Customer Churn Dashboard")

# Load dataset
df = pd.read_csv("data/churn.csv")

# Clean columns
df.columns = df.columns.str.strip()

# KPI Metrics
total_customers = len(df)

churn_count = df["Churn Label"].value_counts().get("Yes", 0)

churn_rate = (churn_count / total_customers) * 100

revenue_risk = df["Monthly Charges"].sum()

# Metrics layout
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)

col2.metric("Churn Rate", f"{churn_rate:.2f}%")

col3.metric("Monthly Revenue", f"${revenue_risk:,.0f}")

# Load CSV data
df = pd.read_csv("data/churn.csv")

df.columns = df.columns.str.strip()

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Churn Distribution
st.subheader("Churn Distribution")
st.subheader("Customer Churn Distribution")

churn_data = df["Churn Label"].value_counts()

st.bar_chart(churn_data)

st.subheader("Revenue Risk Analysis")

high_risk_customers = churn_count

avg_monthly_revenue = df["Monthly Charges"].mean()

estimated_loss = high_risk_customers * avg_monthly_revenue

st.warning(
    f"Estimated Monthly Revenue at Risk: ${estimated_loss:,.2f}"
)

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
        
        
# Feature Importance Section
st.subheader("Top Factors Affecting Churn")

# Get feature importances
importances = model.feature_importances_

feature_names = model_columns

# Create dataframe
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort values
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.invert_yaxis()

st.pyplot(fig)