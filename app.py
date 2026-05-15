import streamlit as st
import pandas as pd
import sqlite3

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