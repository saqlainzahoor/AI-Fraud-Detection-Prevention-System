"""
Precomputes summary statistics and visuals data for the dashboard,
so the Streamlit app doesn't need to reload the full 493MB dataset every time.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from data_preprocessing import load_data, filter_relevant_transactions, chronological_split
from feature_engineering import engineer_features, REALISTIC_FEATURES

DATA_PATH = "../data/raw/PS_20174392719_1491204439457_log.csv"

print("Loading and processing data...")
df = load_data(DATA_PATH)
relevant = filter_relevant_transactions(df)
engineered = engineer_features(relevant)
train_data, test_data = chronological_split(engineered)

model = joblib.load("../models/fraud_model.joblib")

X_test = test_data[REALISTIC_FEATURES]
y_test = test_data['isFraud']
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 1. Fraud by transaction type
fraud_by_type = engineered.groupby('type_TRANSFER')['isFraud'].agg(['sum', 'count', 'mean'])

# 2. Amount distribution summary (fraud vs non-fraud)
amount_summary = engineered.groupby('isFraud')['amount'].describe()

# 3. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 4. PR curve data
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

# Sab kuch ek dictionary mein bundle karo aur save karo
dashboard_data = {
    "total_transactions": len(test_data),
    "fraud_count": int(y_test.sum()),
    "fraud_rate": float(y_test.mean() * 100),
    "confusion_matrix": cm.tolist(),
    "amount_summary": amount_summary.to_dict(),
    "precision_vals": precision_vals[::50].tolist(),  # sample kar ke chota karte hain
    "recall_vals": recall_vals[::50].tolist(),
    "type_fraud_counts": {"TRANSFER": 4097, "CASH_OUT": 4116},
}

joblib.dump(dashboard_data, "../models/dashboard_data.joblib")
print("Dashboard data saved successfully!")
print(dashboard_data)