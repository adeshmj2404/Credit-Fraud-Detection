from fastapi import FastAPI
from backend.app.schemas import FraudRequest
import numpy as np
import joblib

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and scaler
model = joblib.load(r"D:\Project Work\Credit-Fraud-Detection\backend\models\fraud_model.pkl")
scaler = joblib.load(r"D:\Project Work\Credit-Fraud-Detection\backend\models\scaler.pkl")

@app.post("/predict")
def predict(data: FraudRequest):

    # convert incoming request into correct feature order
    features = [
        data.Time, data.V1, data.V2, data.V3, data.V4, data.V5, data.V6,
        data.V7, data.V8, data.V9, data.V10, data.V11, data.V12, data.V13,
        data.V14, data.V15, data.V16, data.V17, data.V18, data.V19, data.V20,
        data.V21, data.V22, data.V23, data.V24, data.V25, data.V26, data.V27,
        data.V28, data.Amount
    ]

    X = np.array(features).reshape(1, -1)

    # scale
    X_scaled = scaler.transform(X)

    # predict
    pred = int(model.predict(X_scaled)[0])

    return {"prediction": pred}
