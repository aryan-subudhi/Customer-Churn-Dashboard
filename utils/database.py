import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("customer_churn.db")

# Load dataset
df = pd.read_csv("data/churn.csv")

# Store data in SQL table
df.to_sql("customers", conn, if_exists="replace", index=False)

print("Database created successfully!")

# Close connection
conn.close()