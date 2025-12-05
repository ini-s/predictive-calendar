import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import pickle
from collections import defaultdict
import calendar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PatternType(Enum):
    MONTHLY = "MONTHLY"
    WEEKLY = "WEEKLY"
    BI_WEEKLY = "BI_WEEKLY"
    QUARTERLY = "QUARTERLY"
    IRREGULAR = "IRREGULAR"


class PredictionCategory(Enum):
    BILL_PAYMENT = "BILL_PAYMENT"
    SUBSCRIPTION = "SUBSCRIPTION"
    LOAN_REPAYMENT = "LOAN_REPAYMENT"
    SAVINGS_CONTRIBUTION = "SAVINGS_CONTRIBUTION"
    INSURANCE_PREMIUM = "INSURANCE_PREMIUM"
    AIRTIME_DATA = "AIRTIME_DATA"
    TRANSFER = "TRANSFER"
    DIRECT_DEBIT = "DIRECT_DEBIT"
    INVESTMENT = "INVESTMENT"
    OTHER = "OTHER"


@dataclass
class DetectedPattern:
    description: str
    category: PredictionCategory
    pattern_type: PatternType
    typical_amount: float
    amount_variance: float
    typical_day_of_month: int
    day_variance: float
    occurrence_count: int
    confidence: float
    source_transaction_ids: List[str] = field(default_factory=list)
    last_occurrence: datetime = None
    receiver_id: str = None  # The merchant/recipient ID that groups these transactions


@dataclass
class Prediction:
    prediction_id: str
    title: str
    description: str
    category: PredictionCategory
    predicted_date: datetime
    predicted_amount: float
    currency: str
    confidence_score: float
    reasoning: str
    pattern_type: PatternType
    source_transaction_ids: List[str]
    reminder_hours_before: int = 24

# Maps transaction types from the database to prediction categories
TRANSACTION_TYPE_TO_CATEGORY = {
    'BillPayment': PredictionCategory.BILL_PAYMENT,
    'DirectDebit': PredictionCategory.DIRECT_DEBIT,
    'LoanRepayment': PredictionCategory.LOAN_REPAYMENT,
    'SavingsContribution': PredictionCategory.SAVINGS_CONTRIBUTION,
    'InsurancePayment': PredictionCategory.INSURANCE_PREMIUM,
    'AirtimeTopup': PredictionCategory.AIRTIME_DATA,
    'DataPurchase': PredictionCategory.AIRTIME_DATA,
    'TransferLocal': PredictionCategory.TRANSFER,
    'TransferInternational': PredictionCategory.TRANSFER,
    'InvestmentPurchase': PredictionCategory.INVESTMENT,
    'VoucherRedeem': PredictionCategory.OTHER,
    'CardIssue': PredictionCategory.OTHER,
    'TravelBooking': PredictionCategory.OTHER,
    'LoanDisbursement': PredictionCategory.OTHER,
}


RECURRING_CATEGORIES = {
    PredictionCategory.BILL_PAYMENT: 0.95,
    PredictionCategory.LOAN_REPAYMENT: 0.98,
    PredictionCategory.SAVINGS_CONTRIBUTION: 0.90,
    PredictionCategory.INSURANCE_PREMIUM: 0.95,
    PredictionCategory.DIRECT_DEBIT: 0.92,
    PredictionCategory.SUBSCRIPTION: 0.90,
}

# RULE-BASED PATTERN DETECTOR
class RuleBasedPatternDetector:    
    def __init__(self, min_occurrences: int = 2, confidence_threshold: float = 0.5):
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold
    
    def detect_patterns(
        self, 
        transactions: pd.DataFrame,
        analysis_months: int = 3
    ) -> List[DetectedPattern]:
        if transactions.empty:
            return []
        
        transactions = transactions.copy()
        transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
        
        cutoff_date = datetime.now() - timedelta(days=analysis_months * 30)
        recent_transactions = transactions[transactions['TransactionDate'] >= cutoff_date]
        
        if recent_transactions.empty:
            return []
        
        patterns = []
        
        receiver_patterns = self._detect_by_receiver_id(recent_transactions)
        patterns.extend(receiver_patterns)
        
        receiver_col = self._get_receiver_column(recent_transactions)
        if receiver_col:
            no_receiver_txs = recent_transactions[
                recent_transactions[receiver_col].isna() |
                (recent_transactions[receiver_col] == '')
            ]
        else:
            no_receiver_txs = recent_transactions
            
        if not no_receiver_txs.empty and len(no_receiver_txs) >= self.min_occurrences:
            fallback_patterns = self._detect_by_type_and_amount(no_receiver_txs)
            patterns.extend(fallback_patterns)
        
        weekly_patterns = self._detect_weekly_patterns(recent_transactions)
        patterns.extend(weekly_patterns)
        
        patterns = self._deduplicate_patterns(patterns)
        
        patterns = [p for p in patterns if p.confidence >= self.confidence_threshold]

        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns
    
    def _get_receiver_column(self, transactions: pd.DataFrame) -> Optional[str]:
        for col_name in ['ReceiverId', 'receiver_id', 'ReceiverID', 'Receiver_Id', 'ReceiverName',]:
            if col_name in transactions.columns:
                return col_name
        return None
    
    def _detect_by_receiver_id(self, transactions: pd.DataFrame) -> List[DetectedPattern]:
        patterns = []

        receiver_col = self._get_receiver_column(transactions)
        
        if receiver_col is None:
            return patterns
        
        valid_transactions = transactions[
            transactions[receiver_col].notna() & 
            (transactions[receiver_col] != '')
        ].copy()
        
        if valid_transactions.empty:
            return patterns
        
        for receiver_id, group in valid_transactions.groupby(receiver_col):
            if len(group) >= self.min_occurrences:
                receiver_name = str(receiver_id)
                
                pattern = self._analyze_group(group, receiver_id=receiver_name)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_by_type_and_amount(self, transactions: pd.DataFrame) -> List[DetectedPattern]:
        patterns = []
        
        for tx_type, group in transactions.groupby('Type'):
            if len(group) < self.min_occurrences:
                continue
            
            if len(group) >= 3:
                amounts = group[['Amount']].values
                
                n_clusters = min(3, len(group) // 2)
                if n_clusters >= 1:
                    try:
                        scaler = StandardScaler()
                        amounts_scaled = scaler.fit_transform(amounts)
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        group = group.copy()
                        group['cluster'] = kmeans.fit_predict(amounts_scaled)
                        
                        for cluster_id in range(n_clusters):
                            cluster_group = group[group['cluster'] == cluster_id]
                            if len(cluster_group) >= self.min_occurrences:
                                pattern = self._analyze_group(
                                    cluster_group, 
                                    f"{tx_type}_cluster_{cluster_id}"
                                )
                                if pattern:
                                    patterns.append(pattern)
                    except Exception:
                        pattern = self._analyze_group(group, tx_type)
                        if pattern:
                            patterns.append(pattern)
            else:
                pattern = self._analyze_group(group, tx_type)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_weekly_patterns(self, transactions: pd.DataFrame) -> List[DetectedPattern]:
        patterns = []
        transactions = transactions.copy()
        transactions['day_of_week'] = transactions['TransactionDate'].dt.dayofweek
        transactions['week'] = transactions['TransactionDate'].dt.isocalendar().week
        
        for tx_type in ['AirtimeTopup', 'DataPurchase', 'SavingsContribution']:
            type_txs = transactions[transactions['Type'] == tx_type]
            if len(type_txs) < 4:
                continue
            
            day_counts = type_txs['day_of_week'].value_counts()
            if len(day_counts) > 0:
                most_common_day = day_counts.index[0]
                day_frequency = day_counts.iloc[0] / len(type_txs)
                
                if day_frequency >= 0.5: 
                    weeks_with_tx = type_txs['week'].nunique()
                    total_weeks = (
                        transactions['TransactionDate'].max() - 
                        transactions['TransactionDate'].min()
                    ).days / 7
                    
                    if total_weeks > 0 and weeks_with_tx / total_weeks >= 0.6:
                        same_day_txs = type_txs[type_txs['day_of_week'] == most_common_day]
                        pattern = self._create_weekly_pattern(same_day_txs, most_common_day)
                        if pattern:
                            patterns.append(pattern)
        
        return patterns
    
    def _analyze_group(
        self, 
        group: pd.DataFrame, 
        receiver_id: str = None
    ) -> Optional[DetectedPattern]:
        if len(group) < self.min_occurrences:
            return None
        
        amounts = group['Amount'].values
        typical_amount = float(np.median(amounts))
        amount_std = float(np.std(amounts)) if len(amounts) > 1 else 0
        amount_variance = amount_std / typical_amount if typical_amount > 0 else 0
        
        days = group['TransactionDate'].dt.day.values
        typical_day = int(np.median(days))
        day_std = float(np.std(days)) if len(days) > 1 else 0

        pattern_type = self._determine_pattern_type(group)
        
        confidence = self._calculate_confidence(
            occurrence_count=len(group),
            amount_variance=amount_variance,
            day_variance=day_std,
            pattern_type=pattern_type,
            transaction_type=group['Type'].iloc[0] if 'Type' in group.columns else None
        )
              
        tx_type = group['Type'].iloc[0] if 'Type' in group.columns else 'Other'
        category = TRANSACTION_TYPE_TO_CATEGORY.get(tx_type, PredictionCategory.OTHER)

        most_recent = group.sort_values('TransactionDate', ascending=False).iloc[0]
        
        if receiver_id:
            description = f"Payment to {receiver_id}"
        else:
            description = most_recent['Description']
        
        return DetectedPattern(
            description=description,
            category=category,
            pattern_type=pattern_type,
            typical_amount=typical_amount,
            amount_variance=amount_variance,
            typical_day_of_month=typical_day,
            day_variance=day_std,
            occurrence_count=len(group),
            confidence=confidence,
            source_transaction_ids=group['Id'].tolist() if 'Id' in group.columns else [],
            last_occurrence=group['TransactionDate'].max(),
            receiver_id=receiver_id
        )
    
    def _create_weekly_pattern(
        self, 
        group: pd.DataFrame, 
        day_of_week: int
    ) -> Optional[DetectedPattern]:
        """Create a pattern for weekly recurring transactions."""
        if len(group) < 3:
            return None
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week]
        
        amounts = group['Amount'].values
        typical_amount = float(np.median(amounts))
        amount_variance = float(np.std(amounts)) / typical_amount if typical_amount > 0 else 0
        
        tx_type = group['Type'].iloc[0]
        category = TRANSACTION_TYPE_TO_CATEGORY.get(tx_type, PredictionCategory.OTHER)
        
        confidence = min(0.85, 0.6 + (len(group) * 0.05))  # Higher with more occurrences
        
        most_recent = group.sort_values('TransactionDate', ascending=False).iloc[0]
        
        return DetectedPattern(
            description=f"Weekly {most_recent['Description']} ({day_name}s)",
            category=category,
            pattern_type=PatternType.WEEKLY,
            typical_amount=typical_amount,
            amount_variance=amount_variance,
            typical_day_of_month=day_of_week, 
            day_variance=0,
            occurrence_count=len(group),
            confidence=confidence,
            source_transaction_ids=group['Id'].tolist() if 'Id' in group.columns else [],
            last_occurrence=group['TransactionDate'].max()
        )
    
    def _determine_pattern_type(self, group: pd.DataFrame) -> PatternType:
        dates = group['TransactionDate'].sort_values()
        if len(dates) < 2:
            return PatternType.IRREGULAR

        gaps = dates.diff().dt.days.dropna()
        if len(gaps) == 0:
            return PatternType.IRREGULAR
        
        median_gap = gaps.median()
        
        if 6 <= median_gap <= 8:
            return PatternType.WEEKLY
        elif 12 <= median_gap <= 16:
            return PatternType.BI_WEEKLY
        elif 25 <= median_gap <= 35:
            return PatternType.MONTHLY
        elif 85 <= median_gap <= 100:
            return PatternType.QUARTERLY
        else:
            return PatternType.IRREGULAR
    
    def _calculate_confidence(
        self,
        occurrence_count: int,
        amount_variance: float,
        day_variance: float,
        pattern_type: PatternType,
        transaction_type: str = None
    ) -> float:
        base_confidence = 0.5
        
        occurrence_bonus = min(0.25, occurrence_count * 0.05)
        
        amount_bonus = max(0, 0.15 - (amount_variance * 0.5))
        
        day_bonus = max(0, 0.1 - (day_variance * 0.01))
        
        pattern_bonus = {
            PatternType.MONTHLY: 0.1,
            PatternType.WEEKLY: 0.08,
            PatternType.BI_WEEKLY: 0.08,
            PatternType.QUARTERLY: 0.05,
            PatternType.IRREGULAR: 0.0,
        }.get(pattern_type, 0)
        
        type_bonus = 0
        if transaction_type:
            category = TRANSACTION_TYPE_TO_CATEGORY.get(transaction_type)
            if category in RECURRING_CATEGORIES:
                type_bonus = RECURRING_CATEGORIES[category] * 0.1
        
        confidence = (
            base_confidence + 
            occurrence_bonus + 
            amount_bonus + 
            day_bonus + 
            pattern_bonus + 
            type_bonus
        )
        
        return min(0.98, max(0.3, confidence))  # Clamp between 0.3 and 0.98
    
    def _deduplicate_patterns(
        self, 
        patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:
        if len(patterns) <= 1:
            return patterns
        
        unique_patterns = []
        
        for pattern in patterns:
            is_duplicate = False
            
            for existing in unique_patterns:
                same_category = pattern.category == existing.category
                
                max_amount = max(float(pattern.typical_amount), float(existing.typical_amount), 1.0)
                amount_diff = abs(float(pattern.typical_amount) - float(existing.typical_amount))
                similar_amount = (amount_diff / max_amount) < 0.1
                
                day_diff = abs(int(pattern.typical_day_of_month) - int(existing.typical_day_of_month))
                similar_day = day_diff <= 3
                
                if same_category and similar_amount and similar_day:
                    if pattern.confidence > existing.confidence:
                        unique_patterns.remove(existing)
                        unique_patterns.append(pattern)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_patterns.append(pattern)
        
        return unique_patterns


# PREDICTION GENERATOR

class PredictionGenerator:
    def __init__(self, max_predictions: int = 10):
        self.max_predictions = max_predictions
        self.id_counter = 0
    
    def generate_predictions(
        self,
        patterns: List[DetectedPattern],
        target_month: str,  # Format: "YYYY-MM"
        user_id: str
    ) -> List[Prediction]:
        if not patterns:
            return []
        
        year, month = map(int, target_month.split('-'))
        
        predictions = []
        
        for pattern in patterns:
            prediction = self._pattern_to_prediction(
                pattern=pattern,
                year=year,
                month=month,
                user_id=user_id
            )
            if prediction:
                predictions.append(prediction)

        predictions.sort(key=lambda x: x.confidence_score, reverse=True)
        predictions = predictions[:self.max_predictions]

        predictions.sort(key=lambda x: x.predicted_date)
        
        return predictions
    
    def _pattern_to_prediction(
        self,
        pattern: DetectedPattern,
        year: int,
        month: int,
        user_id: str
    ) -> Optional[Prediction]:
        self.id_counter += 1
        prediction_id = f"pred_{user_id[:8]}_{year}{month:02d}_{self.id_counter:04d}"
        
        if pattern.pattern_type == PatternType.WEEKLY:
            predicted_date = self._get_next_weekday(year, month, pattern.typical_day_of_month)
        else:
            days_in_month = calendar.monthrange(year, month)[1]
            day = min(pattern.typical_day_of_month, days_in_month)
            predicted_date = datetime(year, month, day)
        
        if predicted_date < datetime.now():
            if pattern.pattern_type == PatternType.MONTHLY:
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                days_in_month = calendar.monthrange(year, month)[1]
                day = min(pattern.typical_day_of_month, days_in_month)
                predicted_date = datetime(year, month, day)
        
        reasoning = self._generate_reasoning(pattern)

        title = self._create_title(pattern)
        
        return Prediction(
            prediction_id=prediction_id,
            title=title,
            description=pattern.description,
            category=pattern.category,
            predicted_date=predicted_date,
            predicted_amount=round(pattern.typical_amount, 2),
            currency="NGN",
            confidence_score=round(pattern.confidence, 3),
            reasoning=reasoning,
            pattern_type=pattern.pattern_type,
            source_transaction_ids=pattern.source_transaction_ids,
            reminder_hours_before=self._get_reminder_hours(pattern.category)
        )
    
    def _get_next_weekday(self, year: int, month: int, day_of_week: int) -> datetime:
        first_day = datetime(year, month, 1)
        day_of_week = int(day_of_week)
        days_until = (day_of_week - first_day.weekday()) % 7
        return first_day + timedelta(days=int(days_until))
    
    def _generate_reasoning(self, pattern: DetectedPattern) -> str:
        freq_text = {
            PatternType.MONTHLY: "monthly",
            PatternType.WEEKLY: "weekly",
            PatternType.BI_WEEKLY: "bi-weekly",
            PatternType.QUARTERLY: "quarterly",
            PatternType.IRREGULAR: "recurring"
        }.get(pattern.pattern_type, "recurring")
        
        return (
            f"Based on {pattern.occurrence_count} {freq_text} transactions "
            f"with an average of ₦{pattern.typical_amount:,.2f}. "
            f"Typically occurs around day {pattern.typical_day_of_month} of the month."
        )
    
    def _create_title(self, pattern: DetectedPattern) -> str:
        desc = pattern.description
        
        for suffix in [' - January', ' - February', ' - March', ' - April', ' - May', 
                       ' - June', ' - July', ' - August', ' - September', ' - October',
                       ' - November', ' - December']:
            desc = desc.replace(suffix, '')
        
        if len(desc) > 50:
            desc = desc[:47] + "..."
        
        return desc
    
    def _get_reminder_hours(self, category: PredictionCategory) -> int:
        reminder_mapping = {
            PredictionCategory.LOAN_REPAYMENT: 48,
            PredictionCategory.BILL_PAYMENT: 24,
            PredictionCategory.INSURANCE_PREMIUM: 48,
            PredictionCategory.SAVINGS_CONTRIBUTION: 12,
            PredictionCategory.DIRECT_DEBIT: 24,
            PredictionCategory.SUBSCRIPTION: 24,
        }
        return reminder_mapping.get(category, 24)

# FEEDBACK LEARNING MODULE

class FeedbackLearner:
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self.user_preferences: Dict[str, Dict] = defaultdict(lambda: {
            'kept_patterns': [],
            'discarded_patterns': [],
            'edited_patterns': [],
            'category_weights': defaultdict(lambda: 1.0),
            'feedback_count': 0
        })
    
    def record_feedback(
        self,
        user_id: str,
        prediction_id: str,
        action: str,  # 'KEPT', 'DISCARDED', 'EDITED'
        prediction_details: Dict,
        edited_data: Dict = None,
        discard_reason: str = None
    ) -> Dict:
        prefs = self.user_preferences[user_id]
        prefs['feedback_count'] += 1
        
        pattern_signature = self._create_pattern_signature(prediction_details)
        
        if action == 'KEPT':
            prefs['kept_patterns'].append({
                'signature': pattern_signature,
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id
            })
            # Boost category confidence
            category = prediction_details.get('category')
            if category:
                prefs['category_weights'][category] = min(1.5, prefs['category_weights'][category] + 0.1)
        
        elif action == 'DISCARDED':
            prefs['discarded_patterns'].append({
                'signature': pattern_signature,
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id,
                'reason': discard_reason
            })
            # Reduce category confidence
            category = prediction_details.get('category')
            if category:
                prefs['category_weights'][category] = max(0.5, prefs['category_weights'][category] - 0.1)
        
        elif action == 'EDITED':
            prefs['edited_patterns'].append({
                'original_signature': pattern_signature,
                'edited_data': edited_data,
                'timestamp': datetime.now().isoformat(),
                'prediction_id': prediction_id
            })
        
        # Check if retraining should be triggered
        retraining_triggered = self._should_trigger_retraining(user_id)
        
        return {
            'processed': True,
            'user_id': user_id,
            'action': action,
            'retraining_triggered': retraining_triggered,
            'total_feedback_count': prefs['feedback_count']
        }
    
    def get_user_adjustments(self, user_id: str) -> Dict:
        prefs = self.user_preferences[user_id]
        
        # Identify frequently discarded patterns
        excluded_signatures = set()
        discard_counts = defaultdict(int)
        
        for discarded in prefs['discarded_patterns']:
            sig = discarded['signature']
            discard_counts[sig] += 1
            if discard_counts[sig] >= 2:  # Discarded twice = exclude
                excluded_signatures.add(sig)
        
        return {
            'category_weights': dict(prefs['category_weights']),
            'excluded_patterns': list(excluded_signatures),
            'total_kept': len(prefs['kept_patterns']),
            'total_discarded': len(prefs['discarded_patterns']),
            'total_edited': len(prefs['edited_patterns'])
        }
    
    def _create_pattern_signature(self, prediction_details: Dict) -> str:
        category = prediction_details.get('category', 'UNKNOWN')
        amount_bucket = int(prediction_details.get('predicted_amount', 0) / 1000) * 1000
        day_bucket = prediction_details.get('predicted_date', datetime.now())
        if isinstance(day_bucket, datetime):
            day_bucket = day_bucket.day
        day_bucket = (day_bucket // 5) * 5  # Group by 5-day ranges
        
        return f"{category}_{amount_bucket}_{day_bucket}"
    
    def _should_trigger_retraining(self, user_id: str) -> bool:
        prefs = self.user_preferences[user_id]
        return prefs['feedback_count'] >= 10 and prefs['feedback_count'] % 10 == 0
    
    def save(self):
        if self.storage_path:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(dict(self.user_preferences), f)
    
    def load(self):
        if self.storage_path:
            try:
                with open(self.storage_path, 'rb') as f:
                    loaded = pickle.load(f)
                    for user_id, prefs in loaded.items():
                        self.user_preferences[user_id] = prefs
            except FileNotFoundError:
                pass  # No existing data, start fresh

# MAIN PREDICTION SERVICE

class PredictiveCalendarService:
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, feedback_storage_path: str = None):
        self.pattern_detector = RuleBasedPatternDetector()
        self.prediction_generator = PredictionGenerator()
        self.feedback_learner = FeedbackLearner(feedback_storage_path)
        
        # Try to load existing feedback data
        self.feedback_learner.load()
    
    def generate_predictions(
        self,
        user_id: str,
        transactions: pd.DataFrame,
        target_month: str,
        max_predictions: int = 10
    ) -> Dict:
        import time
        start_time = time.time()

        user_adjustments = self.feedback_learner.get_user_adjustments(user_id)

        patterns = self.pattern_detector.detect_patterns(transactions)

        excluded = set(user_adjustments['excluded_patterns'])
        patterns = [p for p in patterns if self._get_pattern_signature(p) not in excluded]

        for pattern in patterns:
            category_key = pattern.category.value
            if category_key in user_adjustments['category_weights']:
                pattern.confidence *= user_adjustments['category_weights'][category_key]

        self.prediction_generator.max_predictions = max_predictions
        predictions = self.prediction_generator.generate_predictions(
            patterns=patterns,
            target_month=target_month,
            user_id=user_id
        )
        
        processing_time = int((time.time() - start_time) * 1000)

        return {
            'status': 'SUCCESS' if predictions else 'INSUFFICIENT_DATA',
            'predictions': [self._prediction_to_dict(p) for p in predictions],
            'metadata': {
                'processing_time_ms': processing_time,
                'transactions_analyzed': len(transactions),
                'patterns_detected': len(patterns),
                'model_version': self.MODEL_VERSION,
                'generated_at': datetime.now().isoformat()
            },
            'user_id': user_id,
            'target_month': target_month
        }
    
    def process_feedback(
        self,
        user_id: str,
        feedback_items: List[Dict]
    ) -> Dict:
        processed_count = 0
        failed_items = []
        retraining_triggered = False
        
        for item in feedback_items:
            try:
                result = self.feedback_learner.record_feedback(
                    user_id=user_id,
                    prediction_id=item.get('prediction_id'),
                    action=item.get('action'),
                    prediction_details=item.get('prediction_details', {}),
                    edited_data=item.get('edited_data'),
                    discard_reason=item.get('discard_reason')
                )
                processed_count += 1
                retraining_triggered = retraining_triggered or result['retraining_triggered']
            except Exception as e:
                failed_items.append({
                    'prediction_id': item.get('prediction_id'),
                    'failure_reason': str(e)
                })
        
        self.feedback_learner.save()
        
        return {
            'status': 'SUCCESS' if not failed_items else 'PARTIAL_SUCCESS',
            'processed_count': processed_count,
            'failed_items': failed_items,
            'retraining_triggered': retraining_triggered,
            'message': f"Processed {processed_count} feedback items"
        }
    
    def _prediction_to_dict(self, prediction: Prediction) -> Dict:
        return {
            'prediction_id': prediction.prediction_id,
            'title': prediction.title,
            'description': prediction.description,
            'category': prediction.category.value,
            'predicted_date': prediction.predicted_date.isoformat(),
            'predicted_amount': prediction.predicted_amount,
            'currency': prediction.currency,
            'confidence_score': prediction.confidence_score,
            'reasoning': prediction.reasoning,
            'pattern_type': prediction.pattern_type.value,
            'source_transaction_ids': prediction.source_transaction_ids,
            'reminder_hours_before': prediction.reminder_hours_before
        }
    
    def _get_pattern_signature(self, pattern: DetectedPattern) -> str:
        amount_bucket = int(pattern.typical_amount / 1000) * 1000
        day_bucket = (pattern.typical_day_of_month // 5) * 5
        return f"{pattern.category.value}_{amount_bucket}_{day_bucket}"
