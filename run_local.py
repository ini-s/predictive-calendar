#This runs the prediction service without Azure Functions

import json
from models.prediction_model import PredictiveCalendarService
import pandas as pd

# Initialize the service
service = PredictiveCalendarService()

# Load transactions from the CSV
df = pd.read_csv('data/user_transactions_mapping.csv')

# Filter for one user
user_id = df['UserId'].iloc[0]
user_transactions = df[df['UserId'] == user_id].copy()

# Rename columns to match expected format
user_transactions = user_transactions.rename(columns={
    'TransactionId': 'Id',
    'UserId': 'UserId'
})

print("=" * 60)
print(f"GENERATING PREDICTIONS FOR USER: {user_id[:20]}...")
print(f"Total transactions: {len(user_transactions)}")
print("=" * 60)

# Generate predictions
result = service.generate_predictions(
    user_id=user_id,
    transactions=user_transactions,
    target_month="2025-12",
    max_predictions=10
)

print(f"\nStatus: {result['status']}")
print(f"Processing Time: {result['metadata']['processing_time_ms']}ms")
print(f"Patterns Detected: {result['metadata']['patterns_detected']}")
print(f"\nPredictions ({len(result['predictions'])}):")
print("-" * 60)

for i, pred in enumerate(result['predictions'], 1):
    print(f"\n{i}. {pred['title']}")
    print(f"   Category: {pred['category']}")
    print(f"   Date: {pred['predicted_date'][:10]}")
    print(f"   Amount: ₦{pred['predicted_amount']:,.2f}")
    print(f"   Confidence: {pred['confidence_score']*100:.1f}%")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)