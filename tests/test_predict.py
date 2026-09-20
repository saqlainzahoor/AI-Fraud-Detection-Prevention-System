"""
Basic tests for the fraud detection prediction pipeline.
Run with: python test_predict.py (from tests/ folder)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import predict_transaction, get_risk_level


def test_risk_level_boundaries():
    """Risk level thresholds sahi kaam kar rahe hain ya nahi, check karo."""
    assert get_risk_level(0.1) == ("Low", "Allow")
    assert get_risk_level(0.5) == ("Medium", "Review")
    assert get_risk_level(0.75) == ("High", "Block + Alert")
    assert get_risk_level(0.95) == ("Critical", "Immediate Block + Priority Alert")
    print("PASS: test_risk_level_boundaries")


def test_predict_returns_expected_keys():
    """predict_transaction() sahi structure return kar raha hai ya nahi."""
    sample = {
        'amount': 50000,
        'oldbalanceDest': 10000,
        'newbalanceDest': 60000,
        'type_CASH_OUT': True,
        'type_TRANSFER': False,
        'hour_of_day': 14
    }
    result = predict_transaction(sample)

    assert "fraud_probability" in result
    assert "risk_level" in result
    assert "action" in result
    assert "top_reasons" in result
    assert 0 <= result["fraud_probability"] <= 1
    assert len(result["top_reasons"]) == 3
    print("PASS: test_predict_returns_expected_keys")


def test_high_risk_pattern_detected():
    """Known high-risk pattern (drained destination account) sahi flag hota hai ya nahi."""
    high_risk_sample = {
        'amount': 500000,
        'oldbalanceDest': 0,
        'newbalanceDest': 0,
        'type_CASH_OUT': False,
        'type_TRANSFER': True,
        'hour_of_day': 3
    }
    result = predict_transaction(high_risk_sample)
    assert result["fraud_probability"] > 0.5
    print("PASS: test_high_risk_pattern_detected")


if __name__ == "__main__":
    test_risk_level_boundaries()
    test_predict_returns_expected_keys()
    test_high_risk_pattern_detected()
    print("\nAll tests passed!")