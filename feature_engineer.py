import pandas as pd
from datetime import datetime
import numpy as np
from typing import List
from pydantic import BaseModel


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    sender_id: str
    receiver_id: str
    amount: float
    transaction_date: datetime


class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
            "day_of_month",
            "day_of_week",
            "amount",
            "days_since_last",
            "amount_ratio",
            "is_same_period",
        ]

    def extract_features(
        self, current_tx: Transaction, historical_tx: List[Transaction]
    ) -> pd.DataFrame:
        features = []

        current_date = pd.to_datetime(current_tx.transaction_date)

        for tx in historical_tx:
            tx_date = pd.to_datetime(tx.transaction_date)

            feature_set = {
                "day_of_month": tx_date.day,
                "day_of_week": tx_date.dayofweek,
                "amount": tx.amount,
                "days_since_last": (current_date - tx_date).days,
                "amount_ratio": current_tx.amount / tx.amount if tx.amount != 0 else 1,
                "is_same_period": (
                    1
                    if abs(current_date.normalize() - tx_date.normalize()).days() <= 3
                    else 0
                ),
            }

            features.append(feature_set)

        return pd.DataFrame(features)

    def calculate_temporal_similarity(
        self, current_tx: Transaction, historical_tx: List[Transaction]
    ) -> float:
        if not historical_tx:
            return 0.0

        current_date = pd.to_datetime(current_tx.transaction_date).normalize()
        diffs = []
        for tx in historical_tx:
            tx_date = pd.to_datetime(tx.transaction_date).normalize()
            days_diff_abs = abs((current_date - tx_date).days)
            diffs.append(days_diff_abs)

        similar = sum(1 for d in diffs if d <= 3)
        return similar / len(diffs)