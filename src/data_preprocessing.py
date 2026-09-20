"""
Data preprocessing module for Fraud Detection System.
Handles loading, filtering, and splitting the PaySim dataset.
"""

import pandas as pd


def load_data(filepath):
    """Load raw PaySim dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def filter_relevant_transactions(df):
    """
    Fraud sirf CASH_OUT aur TRANSFER transactions mein hoti hai
    (EDA se confirmed finding). Baqi types ko hata dete hain.
    """
    relevant = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
    return relevant


def chronological_split(df, split_quantile=0.70):
    """
    Time-based split karta hai (random split nahi), kyunke
    fraud rate time ke sath trend dikhati hai (EDA finding).
    """
    split_step = df['step'].quantile(split_quantile)
    train_data = df[df['step'] <= split_step].copy()
    test_data = df[df['step'] > split_step].copy()
    return train_data, test_data


if __name__ == "__main__":
    # Quick test run
    df = load_data("../data/raw/PS_20174392719_1491204439457_log.csv")
    relevant = filter_relevant_transactions(df)
    train, test = chronological_split(relevant)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")