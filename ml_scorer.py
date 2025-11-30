import pandas as pd
from datetime import datetime
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from .feature_engineer import FeatureEngineer, Transaction
from .pattern_detector import PatternDetector
from schemas import TransactionBase
from typing import List, Dict

feature_engineer = FeatureEngineer()
pattern_detector = PatternDetector()


def calculate_ml_score(
    current_tx: TransactionBase, historical_tx: List[TransactionBase]
) -> dict:
    if len(historical_tx) < 2:
        return {
            "transaction_id": current_tx.transaction_id,
            "ml_confidence": 0.0,
            "ml_explanation": "Insufficient historical data",
        }

    # Convert to FeatureEngineer Transaction objects
    current_tx_fe = Transaction(
        transaction_id=current_tx.transaction_id,
        user_id=current_tx.user_id,
        sender_id=current_tx.sender_id,
        receiver_id=current_tx.receiver_id,
        amount=current_tx.amount,
        transaction_date=current_tx.transaction_date,
    )

    historical_tx_fe = [
        Transaction(
            transaction_id=tx.transaction_id,
            user_id=tx.user_id,
            sender_id=tx.sender_id,
            receiver_id=tx.receiver_id,
            amount=tx.amount,
            transaction_date=tx.transaction_date,
        )
        for tx in historical_tx
    ]

    # Extract features
    features_df = feature_engineer.extract_features(current_tx_fe, historical_tx_fe)

    # Fit pattern detector
    pattern_detector.fit(features_df)

    # Calculate current transaction features
    current_date = pd.to_datetime(current_tx.transaction_date)
    current_features = {
        "day_of_month": current_date.day,
        "day_of_week": current_date.dayofweek,
        "amount": current_tx.amount,
        "days_since_last": (
            features_df["days_since_last"].mean() if len(features_df) > 0 else 30
        ),
        "amount_ratio": (
            features_df["amount_ratio"].mean() if len(features_df) > 0 else 1.0
        ),
        "is_same_period": 1,
    }

    # Calculate pattern score
    ml_confidence = pattern_detector.calculate_pattern_score(
        current_features, features_df
    )

    return {
        "transaction_id": current_tx.transaction_id,
        "ml_confidence": ml_confidence,
        "ml_explanation": f"ML pattern similarity: {ml_confidence:.2f}",
    }
