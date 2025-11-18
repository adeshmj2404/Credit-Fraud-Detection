import pandas as pd
import joblib

# Load using joblib (correct)
model = joblib.load(r"D:\Project Work\Credit-Fraud-Detection\backend\models\fraud_model.pkl")
scaler = joblib.load(r"D:\Project Work\Credit-Fraud-Detection\backend\models\scaler.pkl")

# Load dataset
df = pd.read_csv(r"D:\Project Work\Credit-Fraud-Detection\ml\dataset\creditcard.csv")

# Pick the first FRAUD transaction
fraud = df[df["Class"] == 1].iloc[0]

print("\n--- FRAUD TRANSACTION ---")
print(fraud)

# Prepare input (drop Class column)
input_data = fraud.drop("Class")

# Scale
scaled = scaler.transform([input_data])

# Predict
pred = model.predict(scaled)[0]

print("\n--- MODEL PREDICTION ---")
print("Prediction:", pred)

if pred == 1:
    print("⚠️ FRAUD DETECTED — MODEL WORKS!")
else:
    print("❌ WRONG — Model is predicting SAFE for real fraud.")
