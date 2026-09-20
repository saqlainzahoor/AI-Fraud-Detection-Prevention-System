"""
Training pipeline for AI-Based Digital Banking Fraud Detection System.

Final model: Random Forest trained on REALISTIC features only
(balance-drain/error features excluded due to PaySim simulation artifact -
see ablation study results in README/notebooks/02_modeling.ipynb).
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from data_preprocessing import load_data, filter_relevant_transactions, chronological_split
from feature_engineering import engineer_features, REALISTIC_FEATURES


DATA_PATH = "../data/raw/PS_20174392719_1491204439457_log.csv"
MODEL_OUTPUT_PATH = "../models/fraud_model.joblib"
FEATURES_OUTPUT_PATH = "../models/feature_list.joblib"


def train_pipeline():
    print("Step 1/5: Loading data...")
    df = load_data(DATA_PATH)

    print("Step 2/5: Filtering relevant transactions (CASH_OUT, TRANSFER)...")
    relevant = filter_relevant_transactions(df)

    print("Step 3/5: Engineering features...")
    engineered = engineer_features(relevant)

    print("Step 4/5: Chronological train/test split...")
    train_data, test_data = chronological_split(engineered)

    X_train = train_data[REALISTIC_FEATURES]
    y_train = train_data['isFraud']
    X_test = test_data[REALISTIC_FEATURES]
    y_test = test_data['isFraud']

    print("Step 5/5: Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Final Model Performance (Realistic Features) ===")
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1-score:", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_proba), 4))
    print("PR-AUC:", round(average_precision_score(y_test, y_pred_proba), 4))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]}  FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}  TP: {cm[1][1]}")

    # Save model and feature list
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(REALISTIC_FEATURES, FEATURES_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")

    return model


if __name__ == "__main__":
    train_pipeline()