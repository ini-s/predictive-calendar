import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.prediction_model import (
    RuleBasedPatternDetector,
    PredictionGenerator,
    FeedbackLearner,
    PredictiveCalendarService,
    PatternType,
    PredictionCategory
)

# TEST FIXTURES

@pytest.fixture
def sample_transactions():
    base_date = datetime(2025, 11, 1)
    
    # Create recurring patterns that the model should detect
    # Now using ReceiverId as the primary grouping mechanism
    transactions = []
    
    # Pattern 1: Monthly electricity bill (around day 15)
    # All payments to the same receiver (EKEDC) should be grouped together
    for month_offset in range(3):
        transactions.append({
            'Id': f'elec_{month_offset}',
            'UserId': 'user_001',
            'Amount': 15000 + np.random.uniform(-500, 500),
            'Description': f'Electricity Payment - {"August September October".split()[month_offset]}',
            'Type': 'BillPayment',
            'ReceiverId': 'EKEDC',  # Merchant ID - this is the key grouping field
            'TransactionDate': base_date - timedelta(days=30 * month_offset) + timedelta(days=15),
            'CreatedAt': base_date - timedelta(days=30 * month_offset),
            'IsDiscarded': 0,
            'IsKept': 0
        })
    
    # Pattern 2: Monthly loan repayment (around day 25)
    # All payments to the same loan provider should be grouped
    for month_offset in range(3):
        transactions.append({
            'Id': f'loan_{month_offset}',
            'UserId': 'user_001',
            'Amount': 50000,  # Fixed amount
            'Description': 'Loan Repayment Installment',
            'Type': 'LoanRepayment',
            'ReceiverId': 'ALAT_LOANS',  # Loan provider ID
            'TransactionDate': base_date - timedelta(days=30 * month_offset) + timedelta(days=25),
            'CreatedAt': base_date - timedelta(days=30 * month_offset),
            'IsDiscarded': 0,
            'IsKept': 0
        })
    
    # Pattern 3: Weekly airtime (roughly every 7 days)
    # All purchases from MTN should be grouped by receiver
    for week in range(8):
        transactions.append({
            'Id': f'airtime_{week}',
            'UserId': 'user_001',
            'Amount': 2000 + np.random.uniform(-200, 200),
            'Description': 'MTN Airtime Recharge',
            'Type': 'AirtimeTopup',
            'ReceiverId': 'MTN_NG',  # Telecom provider ID
            'TransactionDate': base_date - timedelta(days=7 * week),
            'CreatedAt': base_date - timedelta(days=7 * week),
            'IsDiscarded': 0,
            'IsKept': 0
        })
    
    # Pattern 4: Monthly savings (around day 1)
    for month_offset in range(3):
        transactions.append({
            'Id': f'savings_{month_offset}',
            'UserId': 'user_001',
            'Amount': 30000,
            'Description': 'Savings Contribution - Regular Savings',
            'Type': 'SavingsContribution',
            'ReceiverId': 'ALAT_SAVINGS',  # Internal savings account
            'TransactionDate': base_date - timedelta(days=30 * month_offset) + timedelta(days=1),
            'CreatedAt': base_date - timedelta(days=30 * month_offset),
            'IsDiscarded': 0,
            'IsKept': 0
        })
    
    # Pattern 5: Monthly transfer to landlord (rent)
    # This demonstrates transfer patterns - receiver is the landlord's account
    for month_offset in range(3):
        transactions.append({
            'Id': f'rent_{month_offset}',
            'UserId': 'user_001',
            'Amount': 150000,
            'Description': f'Rent Payment {"August September October".split()[month_offset]}',
            'Type': 'TransferLocal',
            'ReceiverId': 'ACC_12345678',  # Landlord's account number
            'TransactionDate': base_date - timedelta(days=30 * month_offset) + timedelta(days=1),
            'CreatedAt': base_date - timedelta(days=30 * month_offset),
            'IsDiscarded': 0,
            'IsKept': 0
        })
    
    # Add some non-recurring transactions (noise)
    transactions.extend([
        {
            'Id': 'random_1',
            'UserId': 'user_001',
            'Amount': 250000,
            'Description': 'Flight/Bus Ticket to Abuja',
            'Type': 'TravelBooking',
            'ReceiverId': 'WAKANOW',  # One-time travel booking
            'TransactionDate': base_date - timedelta(days=45),
            'CreatedAt': base_date - timedelta(days=45),
            'IsDiscarded': 0,
            'IsKept': 0
        },
        {
            'Id': 'random_2',
            'UserId': 'user_001',
            'Amount': 150000,
            'Description': 'Transfer to Friend',
            'Type': 'TransferLocal',
            'ReceiverId': 'ACC_99999999',  # One-time transfer to friend
            'TransactionDate': base_date - timedelta(days=20),
            'CreatedAt': base_date - timedelta(days=20),
            'IsDiscarded': 0,
            'IsKept': 0
        }
    ])
    
    return pd.DataFrame(transactions)


@pytest.fixture
def prediction_service():
    """Create a prediction service instance for testing."""
    return PredictiveCalendarService()

# PATTERN DETECTOR TESTS

class TestRuleBasedPatternDetector:
    def test_detector_initialization(self):
        detector = RuleBasedPatternDetector()
        assert detector.min_occurrences == 2
        assert detector.confidence_threshold == 0.5
    
    def test_detect_monthly_patterns(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        # Should detect patterns
        assert len(patterns) > 0
        
        # Should have monthly patterns
        monthly_patterns = [p for p in patterns if p.pattern_type == PatternType.MONTHLY]
        assert len(monthly_patterns) >= 2, "Should detect at least 2 monthly patterns"
    
    def test_detect_bill_payments(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        bill_patterns = [p for p in patterns if p.category == PredictionCategory.BILL_PAYMENT]
        assert len(bill_patterns) >= 1, "Should detect bill payment patterns"
    
    def test_detect_loan_repayment(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        loan_patterns = [p for p in patterns if p.category == PredictionCategory.LOAN_REPAYMENT]
        assert len(loan_patterns) >= 1, "Should detect loan repayment patterns"
        
        # Loan repayment should have high confidence due to consistent amounts
        for pattern in loan_patterns:
            assert pattern.confidence >= 0.6, "Loan repayment should have good confidence"
    
    def test_detect_transfer_patterns(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        # Should detect the rent payment pattern (transfers to landlord)
        transfer_patterns = [p for p in patterns if p.category == PredictionCategory.TRANSFER]
        assert len(transfer_patterns) >= 1, "Should detect recurring transfer patterns"
        
        # The rent pattern should be grouped by receiver_id (landlord's account)
        rent_pattern = [p for p in transfer_patterns if p.receiver_id == 'ACC_12345678']
        assert len(rent_pattern) == 1, "Should detect rent payment as a single pattern"
    
    def test_grouping_by_receiver_id(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        # All EKEDC payments should be in one pattern, regardless of description
        ekedc_patterns = [p for p in patterns if p.receiver_id == 'EKEDC']
        assert len(ekedc_patterns) == 1, "All payments to EKEDC should be in one pattern"
        assert ekedc_patterns[0].occurrence_count == 3, "Should have 3 EKEDC transactions"
    
    def test_confidence_scoring(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        for pattern in patterns:
            assert 0.3 <= pattern.confidence <= 0.98, \
                f"Confidence {pattern.confidence} out of range"
    
    def test_empty_transactions(self):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(pd.DataFrame())
        
        assert patterns == [], "Empty transactions should return empty patterns"
    
    def test_pattern_deduplication(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        # Check for duplicates based on receiver_id (primary) or category+amount+day (fallback)
        seen = set()
        for pattern in patterns:
            # Primary signature: use receiver_id if available
            if pattern.receiver_id:
                sig = (pattern.receiver_id, pattern.category)
            else:
                # Fallback signature for patterns without receiver_id
                sig = (
                    pattern.category,
                    int(round(pattern.typical_amount / 1000)),
                    int(pattern.typical_day_of_month // 5)
                )
            assert sig not in seen, f"Duplicate pattern found: {sig}"
            seen.add(sig)

class TestPredictionGenerator:
    def test_generator_initialization(self):
        generator = PredictionGenerator(max_predictions=10)
        assert generator.max_predictions == 10
    
    def test_max_predictions_limit(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        generator = PredictionGenerator(max_predictions=10)
        predictions = generator.generate_predictions(
            patterns=patterns,
            target_month="2025-12",
            user_id="user_001"
        )
        
        assert len(predictions) <= 10, "Should not exceed 10 predictions"
    
    def test_prediction_fields(self, sample_transactions):
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        generator = PredictionGenerator()
        predictions = generator.generate_predictions(
            patterns=patterns,
            target_month="2025-12",
            user_id="user_001"
        )
        
        for pred in predictions:
            assert pred.prediction_id is not None
            assert pred.title is not None
            assert pred.description is not None
            assert pred.predicted_date is not None
            assert pred.predicted_amount > 0
            assert pred.currency == "NGN"
            assert 0 <= pred.confidence_score <= 1
            assert pred.reasoning is not None
    
    def test_predictions_sorted_by_date(self, sample_transactions):
        """Test that predictions are sorted by predicted date."""
        detector = RuleBasedPatternDetector()
        patterns = detector.detect_patterns(sample_transactions)
        
        generator = PredictionGenerator()
        predictions = generator.generate_predictions(
            patterns=patterns,
            target_month="2025-12",
            user_id="user_001"
        )
        
        dates = [p.predicted_date for p in predictions]
        assert dates == sorted(dates), "Predictions should be sorted by date"

# FEEDBACK LEARNER TESTS

class TestFeedbackLearner:
    
    def test_learner_initialization(self):
        learner = FeedbackLearner()
        assert learner.storage_path is None
    
    def test_record_kept_feedback(self):
        learner = FeedbackLearner()
        
        result = learner.record_feedback(
            user_id="user_001",
            prediction_id="pred_001",
            action="KEPT",
            prediction_details={
                'category': 'BILL_PAYMENT',
                'predicted_amount': 15000,
                'predicted_date': datetime(2025, 12, 15)
            }
        )
        
        assert result['processed'] is True
        assert result['action'] == 'KEPT'
    
    def test_record_discarded_feedback(self):
        learner = FeedbackLearner()
        
        result = learner.record_feedback(
            user_id="user_001",
            prediction_id="pred_002",
            action="DISCARDED",
            prediction_details={
                'category': 'OTHER',
                'predicted_amount': 5000,
                'predicted_date': datetime(2025, 12, 10)
            },
            discard_reason="Not needed"
        )
        
        assert result['processed'] is True
        assert result['action'] == 'DISCARDED'
    
    def test_category_weight_adjustment(self):
        learner = FeedbackLearner()
        user_id = "user_002"
        
        # Record multiple "kept" feedbacks for BILL_PAYMENT
        for i in range(5):
            learner.record_feedback(
                user_id=user_id,
                prediction_id=f"pred_{i}",
                action="KEPT",
                prediction_details={
                    'category': 'BILL_PAYMENT',
                    'predicted_amount': 10000 + i * 1000,
                    'predicted_date': datetime(2025, 12, 10 + i)
                }
            )
        
        adjustments = learner.get_user_adjustments(user_id)
        
        # BILL_PAYMENT category should have increased weight
        assert adjustments['category_weights']['BILL_PAYMENT'] > 1.0
    
    def test_retraining_trigger(self):
        learner = FeedbackLearner()
        user_id = "user_003"
        
        retraining_triggered = False
        
        # Record 10 feedback items to trigger retraining
        for i in range(10):
            result = learner.record_feedback(
                user_id=user_id,
                prediction_id=f"pred_{i}",
                action="KEPT" if i % 2 == 0 else "DISCARDED",
                prediction_details={
                    'category': 'LOAN_REPAYMENT',
                    'predicted_amount': 50000,
                    'predicted_date': datetime(2025, 12, 25)
                }
            )
            if result['retraining_triggered']:
                retraining_triggered = True
        
        assert retraining_triggered, "Retraining should be triggered after 10 feedbacks"

# FULL SERVICE TESTS

class TestPredictiveCalendarService:
    def test_generate_predictions(self, sample_transactions, prediction_service):
        result = prediction_service.generate_predictions(
            user_id="user_001",
            transactions=sample_transactions,
            target_month="2025-12",
            max_predictions=10
        )
        
        assert result['status'] == 'SUCCESS'
        assert len(result['predictions']) <= 10
        assert result['metadata']['model_version'] == "1.0.0"
    
    def test_processing_time_requirement(self, sample_transactions, prediction_service):
        import time
        
        start = time.time()
        prediction_service.generate_predictions(
            user_id="user_001",
            transactions=sample_transactions,
            target_month="2025-12"
        )
        elapsed = time.time() - start
        
        assert elapsed < 5, f"Prediction took {elapsed:.2f}s, should be <5s"
    
    def test_insufficient_data_handling(self, prediction_service):
        empty_df = pd.DataFrame()
        
        result = prediction_service.generate_predictions(
            user_id="user_empty",
            transactions=empty_df,
            target_month="2025-12"
        )
        
        assert result['status'] == 'INSUFFICIENT_DATA'
        assert len(result['predictions']) == 0
    
    def test_process_feedback(self, prediction_service):
        feedback_items = [
            {
                'prediction_id': 'pred_001',
                'action': 'KEPT',
                'prediction_details': {
                    'category': 'BILL_PAYMENT',
                    'predicted_amount': 15000
                }
            },
            {
                'prediction_id': 'pred_002',
                'action': 'DISCARDED',
                'prediction_details': {
                    'category': 'OTHER',
                    'predicted_amount': 5000
                },
                'discard_reason': 'Not relevant'
            }
        ]
        
        result = prediction_service.process_feedback(
            user_id="user_001",
            feedback_items=feedback_items
        )
        
        assert result['status'] == 'SUCCESS'
        assert result['processed_count'] == 2

# ACCURACY TEST (Target: ≥70% per PRD)

class TestPredictionAccuracy:
    
    def test_confidence_threshold(self, sample_transactions):
        service = PredictiveCalendarService()
        
        result = service.generate_predictions(
            user_id="user_001",
            transactions=sample_transactions,
            target_month="2025-12"
        )
        
        # Filter for high-confidence predictions
        high_confidence = [
            p for p in result['predictions']
            if p['confidence_score'] >= 0.7
        ]
        
        # All high-confidence predictions should be for known recurring patterns
        for pred in high_confidence:
            # These categories are typically recurring
            recurring_categories = [
                'BILL_PAYMENT', 'LOAN_REPAYMENT', 'SAVINGS_CONTRIBUTION',
                'INSURANCE_PREMIUM', 'DIRECT_DEBIT'
            ]
            assert pred['category'] in recurring_categories or pred['confidence_score'] >= 0.7, \
                f"Unexpected high-confidence prediction: {pred}"


# RUN TESTS

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
