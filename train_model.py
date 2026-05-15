import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/churn.csv")

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df.dropna(inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "churn_model.pkl")

print("Model trained successfully!")