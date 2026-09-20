"""
FastAPI backend for AI-Based Digital Banking Fraud Detection System.
Exposes a /predict endpoint for real-time transaction risk scoring.
"""

import sys
import os

# src folder ko path mein add karo taake predict.py import ho sake
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_transaction

app = FastAPI(
    title="AI-Based Digital Banking Fraud Detection & Prevention System",
    description="Real-time transaction risk scoring API",
    version="1.0"
)


class Transaction(BaseModel):
    amount: float
    oldbalanceDest: float
    newbalanceDest: float
    type_CASH_OUT: bool
    type_TRANSFER: bool
    hour_of_day: int


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running. Use POST /predict to score a transaction."}


@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_transaction(transaction.model_dump())
    return result