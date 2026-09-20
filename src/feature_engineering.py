"""
Feature engineering module for Fraud Detection System.
Creates behavioral and derived features from raw transaction data.

NOTE: balance_drained_orig aur errorBalanceOrig/Dest features ablation study
mein PaySim simulator ka ek "artifact" pattern nikli thin (bohat strong
lekin possibly non-generalizable signal). Ye features yahan is liye rakhi
gayi hain taake dono model versions (full aur realistic) generate ki ja
sakein - lekin train.py mein hum sirf REALISTIC features use karenge
final model ke liye.
"""

import pandas as pd


def engineer_features(df):
    """
    Raw transaction data se ML-ready features banata hai.
    Input: filtered dataframe (sirf CASH_OUT aur TRANSFER)
    Output: dataframe with engineered features
    """
    data = df.copy()

    # Balance drain indicator (simulation artifact - documented limitation)
    data['balance_drained_orig'] = (
        (data['oldbalanceOrg'] > 0) & (data['newbalanceOrig'] == 0)
    ).astype(int)

    # Balance error features (simulation artifact - documented limitation)
    data['errorBalanceOrig'] = data['oldbalanceOrg'] - data['amount'] - data['newbalanceOrig']
    data['errorBalanceDest'] = data['newbalanceDest'] - data['oldbalanceDest'] - data['amount']

    # One-hot encoding for transaction type
    data = pd.get_dummies(data, columns=['type'], prefix='type')

    # Time-based feature
    data['hour_of_day'] = data['step'] % 24

    # Amount to balance ratio
    data['amount_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)

    return data


# Features jo FINAL model use karega (realistic, non-leakage-prone)
REALISTIC_FEATURES = [
    'amount', 'oldbalanceDest', 'newbalanceDest',
    'type_CASH_OUT', 'type_TRANSFER', 'hour_of_day'
]

# Sab engineered features (reference/comparison ke liye, ablation study mein use hui)
ALL_FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'balance_drained_orig', 'errorBalanceOrig',
    'errorBalanceDest', 'type_CASH_OUT', 'type_TRANSFER',
    'hour_of_day', 'amount_to_balance_ratio'
]


if __name__ == "__main__":
    from data_preprocessing import load_data, filter_relevant_transactions

    df = load_data("../data/raw/PS_20174392719_1491204439457_log.csv")
    relevant = filter_relevant_transactions(df)
    engineered = engineer_features(relevant)

    print("Engineered shape:", engineered.shape)
    print("\nRealistic features:", REALISTIC_FEATURES)
    print("\nSample:\n", engineered[REALISTIC_FEATURES].head())