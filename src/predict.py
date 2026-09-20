"""
Prediction pipeline: takes transaction features, returns fraud probability,
risk level, recommended action, and top contributing reasons.
"""

import os
import joblib
import shap
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_list.joblib")

# Model aur features ek baar load karo (module import hote hi)
_model = joblib.load(MODEL_PATH)
_features = joblib.load(FEATURES_PATH)
_explainer = shap.TreeExplainer(_model)


def get_risk_level(probability):
    """
    Evidence-based risk bands (Step 20 threshold analysis se derive kiye gaye).
    """
    if probability < 0.3:
        return "Low", "Allow"
    elif probability < 0.7:
        return "Medium", "Review"
    elif probability < 0.9:
        return "High", "Block + Alert"
    else:
        return "Critical", "Immediate Block + Priority Alert"


def predict_transaction(transaction_dict):
    """
    transaction_dict: dict with keys matching _features
    e.g. {'amount': 50000, 'oldbalanceDest': 0, 'newbalanceDest': 0,
          'type_CASH_OUT': False, 'type_TRANSFER': True, 'hour_of_day': 2}
    """
    X = pd.DataFrame([transaction_dict])[_features]

    proba = _model.predict_proba(X)[:, 1][0]
    risk_level, action = get_risk_level(proba)

    # SHAP explanation
    shap_vals = _explainer.shap_values(X)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
    row_shap = shap_vals[0]

    feature_impact = list(zip(_features, row_shap, X.iloc[0].values))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, impact, value in feature_impact[:3]:
        direction = "increased" if impact > 0 else "decreased"
        reasons.append(f"{feat} = {value} ({direction} risk)")

    return {
        "fraud_probability": round(float(proba), 4),
        "risk_level": risk_level,
        "action": action,
        "top_reasons": reasons
    }


if __name__ == "__main__":
    # Test transaction
    sample_transaction = {
        'amount': 450000,
        'oldbalanceDest': 0,
        'newbalanceDest': 0,
        'type_CASH_OUT': False,
        'type_TRANSFER': True,
        'hour_of_day': 3
    }

    result = predict_transaction(sample_transaction)
    print(result)