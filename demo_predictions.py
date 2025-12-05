"""
demo_predictions.py
===================
Demonstration script for ALAT Predictive Calendar

This script demonstrates the full prediction pipeline using the actual
transaction data from the CSV files. It can be used to:
1. Validate the model works correctly
2. Generate sample predictions for documentation
3. Test the feedback system
4. Benchmark performance

Usage:
    python demo_predictions.py
"""

import pandas as pd
import json
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.prediction_model import PredictiveCalendarService
from utils.grpc_service import PredictiveCalendarServicer


def load_test_data():
    """Load the test data from CSV files."""
    possible_paths = [
        '/home/claude/predictive_calendar/data/user_transactions_mapping.csv',
        'data/user_transactions_mapping.csv',
        '../data/user_transactions_mapping.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Loading data from: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError("Could not find user_transactions_mapping.csv")


def format_currency(amount):
    """Format amount as Nigerian Naira."""
    return f"₦{amount:,.2f}"


def demo_pattern_detection(df: pd.DataFrame, user_id: str):
    """Demonstrate pattern detection for a single user."""
    print("\n" + "="*70)
    print(f"PATTERN DETECTION DEMO - User: {user_id}")
    print("="*70)
    
    user_txns = df[df['UserId'] == user_id].copy()
    print(f"\nAnalyzing {len(user_txns)} transactions...")
    
    print("\nTransaction Types:")
    type_counts = user_txns['Type'].value_counts()
    for tx_type, count in type_counts.items():
        print(f"  • {tx_type}: {count} transactions")
    
    return user_txns


def demo_prediction_generation(service: PredictiveCalendarService, user_txns: pd.DataFrame, user_id: str):
    """Demonstrate prediction generation."""
    print("\n" + "-"*70)
    print("GENERATING PREDICTIONS")
    print("-"*70)
    
    # Prepare transaction data in the expected format
    txn_data = user_txns.rename(columns={
        'TransactionId': 'Id',
        'UserId': 'user_id',
        'TransactionDate': 'TransactionDate'
    })
    
    # Time the prediction generation
    start_time = time.time()
    
    result = service.generate_predictions(
        user_id=user_id,
        transactions=txn_data,
        target_month="2025-12",
        max_predictions=10
    )
    
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"\n✓ Prediction generated in {elapsed_time:.2f}ms")
    print(f"  Status: {result['status']}")
    print(f"  Patterns detected: {result['metadata']['patterns_detected']}")
    print(f"  Predictions generated: {len(result['predictions'])}")
    
    if result['predictions']:
        print("\n" + "-"*70)
        print("PREDICTED CALENDAR ITEMS FOR DECEMBER 2025")
        print("-"*70)
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"\n{i}. {pred['title']}")
            print(f"   Category: {pred['category']}")
            print(f"   Date: {pred['predicted_date'][:10]}")
            print(f"   Amount: {format_currency(pred['predicted_amount'])}")
            print(f"   Confidence: {pred['confidence_score']*100:.1f}%")
            print(f"   Pattern: {pred['pattern_type']}")
            print(f"   Reasoning: {pred['reasoning']}")
    
    return result


def demo_feedback_submission(service: PredictiveCalendarService, predictions: list, user_id: str):
    """Demonstrate the feedback submission process."""
    print("\n" + "="*70)
    print("FEEDBACK SUBMISSION DEMO")
    print("="*70)
    
    if not predictions:
        print("\nNo predictions to provide feedback on.")
        return
    
    # Simulate user feedback (keep first 2, discard 3rd, edit 4th if exists)
    feedback_items = []
    
    if len(predictions) >= 1:
        feedback_items.append({
            'prediction_id': predictions[0]['prediction_id'],
            'action': 'KEPT',
            'prediction_details': {
                'category': predictions[0]['category'],
                'predicted_amount': predictions[0]['predicted_amount'],
                'predicted_date': predictions[0]['predicted_date']
            }
        })
        print(f"\n✓ KEPT: {predictions[0]['title']}")
    
    if len(predictions) >= 2:
        feedback_items.append({
            'prediction_id': predictions[1]['prediction_id'],
            'action': 'KEPT',
            'prediction_details': {
                'category': predictions[1]['category'],
                'predicted_amount': predictions[1]['predicted_amount'],
                'predicted_date': predictions[1]['predicted_date']
            }
        })
        print(f"✓ KEPT: {predictions[1]['title']}")
    
    if len(predictions) >= 3:
        feedback_items.append({
            'prediction_id': predictions[2]['prediction_id'],
            'action': 'DISCARDED',
            'discard_reason': 'I no longer need this',
            'prediction_details': {
                'category': predictions[2]['category'],
                'predicted_amount': predictions[2]['predicted_amount'],
                'predicted_date': predictions[2]['predicted_date']
            }
        })
        print(f"✗ DISCARDED: {predictions[2]['title']}")
    
    if len(predictions) >= 4:
        feedback_items.append({
            'prediction_id': predictions[3]['prediction_id'],
            'action': 'EDITED',
            'edited_data': {
                'corrected_title': predictions[3]['title'] + ' (corrected)',
                'corrected_amount': predictions[3]['predicted_amount'] + 1000,
                'corrected_date': predictions[3]['predicted_date'],
                'corrected_category': predictions[3]['category']
            },
            'prediction_details': {
                'category': predictions[3]['category'],
                'predicted_amount': predictions[3]['predicted_amount'],
                'predicted_date': predictions[3]['predicted_date']
            }
        })
        print(f"✎ EDITED: {predictions[3]['title']}")
    
    # Submit feedback
    result = service.process_feedback(
        user_id=user_id,
        feedback_items=feedback_items
    )
    
    print(f"\nFeedback Result:")
    print(f"  Status: {result['status']}")
    print(f"  Processed: {result['processed_count']} items")
    print(f"  Retraining triggered: {result['retraining_triggered']}")
    
    return result


def demo_grpc_servicer(df: pd.DataFrame, user_id: str):
    """Demonstrate the gRPC servicer (HTTP-compatible interface)."""
    print("\n" + "="*70)
    print("gRPC SERVICER DEMO (HTTP-Compatible Interface)")
    print("="*70)
    
    servicer = PredictiveCalendarServicer()
    
    # Prepare request in proto-compatible format
    user_txns = df[df['UserId'] == user_id].copy()
    
    transactions_list = []
    for _, row in user_txns.iterrows():
        transactions_list.append({
            'id': row['TransactionId'],
            'user_id': row['UserId'],
            'amount': row['Amount'],
            'description': row['Description'],
            'type': row['Type'],
            'transaction_date': str(row['TransactionDate']),
            'created_at': str(row['CreatedAt'])
        })
    
    request = {
        'user_id': user_id,
        'target_month': '2025-12',
        'historical_transactions': transactions_list,
        'max_predictions': 10,
        'timezone': 'Africa/Lagos'
    }
    
    print("\nCalling GeneratePredictions via servicer...")
    response = servicer.GeneratePredictions(request)
    
    print(f"\nResponse Status: {response['status']}")
    print(f"Predictions Count: {len(response['predictions'])}")
    
    if response['metadata']:
        print(f"Processing Time: {response['metadata']['processing_time_ms']}ms")
        print(f"Model Version: {response['metadata']['model_version']}")
    
    # Health check
    print("\nCalling GetPredictionHealth...")
    health = servicer.GetPredictionHealth({'include_diagnostics': True})
    print(f"Service Status: {health['status']}")
    print(f"Model Ready: {health['model_health']['is_ready']}")
    print(f"Uptime: {health['uptime_seconds']} seconds")
    
    return response


def demo_json_output(result: dict):
    """Show sample JSON output for backend team."""
    print("\n" + "="*70)
    print("SAMPLE JSON OUTPUT (for Backend Team)")
    print("="*70)
    
    # Create a sample with just 2 predictions for readability
    sample = {
        'status': result['status'],
        'predictions': result['predictions'][:2] if result['predictions'] else [],
        'metadata': result['metadata'],
        'error_message': None
    }
    
    print("\n" + json.dumps(sample, indent=2, default=str))


def run_performance_benchmark(service: PredictiveCalendarService, df: pd.DataFrame):
    """Run a simple performance benchmark."""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    users = df['UserId'].unique()
    times = []
    
    print(f"\nRunning predictions for {len(users)} users...")
    
    for user_id in users:
        user_txns = df[df['UserId'] == user_id].copy()
        user_txns = user_txns.rename(columns={'TransactionId': 'Id'})
        
        start = time.time()
        service.generate_predictions(
            user_id=user_id,
            transactions=user_txns,
            target_month="2025-12"
        )
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  PRD Requirement: <5000ms")
    print(f"  Status: {'✓ PASSED' if max_time < 5000 else '✗ FAILED'}")


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("ALAT PREDICTIVE CALENDAR - DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows the full prediction pipeline including:")
    print("  1. Loading transaction data")
    print("  2. Pattern detection")
    print("  3. Prediction generation")
    print("  4. Feedback submission")
    print("  5. gRPC servicer interface")
    print("  6. Performance benchmarking")
    
    # Load data
    try:
        df = load_test_data()
        print(f"✓ Loaded {len(df)} transaction records")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return
    
    # Initialize service
    service = PredictiveCalendarService()
    print("✓ Prediction service initialized")
    
    # Get first user for demo
    users = df['UserId'].unique()
    demo_user = users[0]
    user_name = df[df['UserId'] == demo_user]['Username'].iloc[0]
    
    print(f"\nDemo user: {user_name} ({demo_user[:8]}...)")
    
    # Run demonstrations
    user_txns = demo_pattern_detection(df, demo_user)
    result = demo_prediction_generation(service, user_txns, demo_user)
    
    if result['predictions']:
        demo_feedback_submission(service, result['predictions'], demo_user)
    
    demo_grpc_servicer(df, demo_user)
    demo_json_output(result)
    run_performance_benchmark(service, df)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the generated predictions")
    print("  2. Share proto file with backend team")
    print("  3. Deploy to Azure Functions")
    print("  4. Test integration with backend")


if __name__ == '__main__':
    main()
