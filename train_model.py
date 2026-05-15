import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/churn.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns
drop_cols = [
    "CustomerID",
    "Churn Label",
    "Churn Reason",
    "Churn Category"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Convert target column
df["Churn Value"] = df["Churn Value"].astype(int)

# Encode categorical columns safely
for column in df.select_dtypes(include=["object"]).columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column].astype(str))
    
# Ensure all columns are numeric
df = df.apply(pd.to_numeric)

# Features and target
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

# Train-test split
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

# Save columns
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model trained successfully!")