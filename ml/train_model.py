import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier

from preprocess import load_and_preprocess
    
print("Loading dataset...")
X, y, scaler = load_and_preprocess(r"D:\Project Work\Credit-Fraud-Detection\ml\dataset\creditcard.csv")
print("Dataset loaded:", X.shape)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Applying SMOTE...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("SMOTE complete:", X_train_res.shape)

print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

print("Evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))

print("Saving model...")
joblib.dump(model, r"D:\Project Work\Credit-Fraud-Detection\backend\models\fraud_model.pkl")
joblib.dump(scaler, r"D:\Project Work\Credit-Fraud-Detection\backend\models\scaler.pkl")

print("Model trained & saved successfully!")
