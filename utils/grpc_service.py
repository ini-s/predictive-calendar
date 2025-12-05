import grpc
from concurrent import futures
from datetime import datetime
from typing import Dict, List, Optional
import json
import sys
import os

#Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import the prediction model
from models.prediction_model import PredictiveCalendarService

#In production, the generated protobuf classes would be imported:
#from protos import predictive_calendar_pb2
#from protos import predictive_calendar_pb2_grpc
# For now, we'll work with dictionaries that match the proto structure


class PredictiveCalendarServicer: 
    def __init__(self, prediction_service: PredictiveCalendarService = None):
        self.service = prediction_service or PredictiveCalendarService()
        self.start_time = datetime.now()
        self.prediction_count = 0
        self.last_prediction_time = None
    
    def GeneratePredictions(self, request: Dict, context=None) -> Dict:
        try:
            #Validate request
            user_id = request.get('user_id')
            if not user_id:
                return self._error_response('INVALID_REQUEST', 'user_id is required')
            
            target_month = request.get('target_month')
            if not target_month or not self._validate_month_format(target_month):
                return self._error_response(
                    'INVALID_REQUEST', 
                    'target_month must be in YYYY-MM format'
                )
            
            #Get transactions (from request or database)
            transactions = request.get('historical_transactions', [])
            
            if not transactions:
                return self._error_response(
                    'INSUFFICIENT_DATA',
                    'No transaction history provided. Please include historical_transactions.'
                )
            
            #Convert transactions to DataFrame
            import pandas as pd
            transactions_df = pd.DataFrame(transactions)
            
            #Rename columns to match expected format if needed
            column_mapping = {
                'id': 'Id',
                'user_id': 'UserId',
                'amount': 'Amount',
                'description': 'Description',
                'type': 'Type',
                'transaction_date': 'TransactionDate',
                'created_at': 'CreatedAt'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in transactions_df.columns and new_col not in transactions_df.columns:
                    transactions_df = transactions_df.rename(columns={old_col: new_col})
            
            # Generate predictions
            max_predictions = request.get('max_predictions', 10)
            result = self.service.generate_predictions(
                user_id=user_id,
                transactions=transactions_df,
                target_month=target_month,
                max_predictions=min(max_predictions, 10)  # Enforce ≤10 limit
            )
            
            # Update stats
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Convert to proto-compatible response
            return self._format_prediction_response(result)
        
        except Exception as e:
            return self._error_response('FAILURE', str(e))
    
    def SubmitFeedback(self, request: Dict, context=None) -> Dict:
        try:
            user_id = request.get('user_id')
            if not user_id:
                return self._feedback_error('INVALID_REQUEST', 'user_id is required')
            
            feedback_items = request.get('feedback_items', [])
            if not feedback_items:
                return self._feedback_error('INVALID_REQUEST', 'feedback_items is required')
            
            # Validate and transform feedback items
            processed_items = []
            for item in feedback_items:
                action = item.get('action', '').upper()
                if action not in ['KEPT', 'DISCARDED', 'EDITED']:
                    continue  # Skip invalid actions
                
                processed_item = {
                    'prediction_id': item.get('prediction_id'),
                    'action': action,
                    'prediction_details': item.get('prediction_details', {}),
                    'discard_reason': item.get('discard_reason')
                }
                
                # Handle edited data
                if action == 'EDITED' and 'edited_data' in item:
                    edited = item['edited_data']
                    processed_item['edited_data'] = {
                        'corrected_title': edited.get('corrected_title'),
                        'corrected_date': edited.get('corrected_date'),
                        'corrected_amount': edited.get('corrected_amount'),
                        'corrected_category': edited.get('corrected_category')
                    }
                
                processed_items.append(processed_item)
            
            # Process feedback
            result = self.service.process_feedback(
                user_id=user_id,
                feedback_items=processed_items
            )
            
            return {
                'status': result['status'],
                'processed_count': result['processed_count'],
                'failed_items': result['failed_items'],
                'message': result['message'],
                'retraining_triggered': result['retraining_triggered']
            }
        
        except Exception as e:
            return self._feedback_error('FAILURE', str(e))
    
    def _validate_month_format(self, month_str: str) -> bool:
        """Validate that month is in YYYY-MM format."""
        try:
            datetime.strptime(month_str, '%Y-%m')
            return True
        except ValueError:
            return False
    
    def _format_prediction_response(self, result: Dict) -> Dict:
        """Format the prediction result to match proto response structure."""
        predictions = []
        
        for pred in result.get('predictions', []):
            predictions.append({
                'prediction_id': pred['prediction_id'],
                'title': pred['title'],
                'description': pred['description'],
                'category': pred['category'],
                'predicted_date': pred['predicted_date'],
                'predicted_amount': int(pred['predicted_amount']),  # Proto uses int64
                'currency': pred['currency'],
                'confidence_score': pred['confidence_score'],
                'reasoning': pred['reasoning'],
                'pattern_type': pred['pattern_type'],
                'source_transaction_ids': pred['source_transaction_ids'],
                'reminder_hours_before': pred['reminder_hours_before']
            })
        
        return {
            'status': result['status'],
            'predictions': predictions,
            'metadata': {
                'processing_time_ms': result['metadata']['processing_time_ms'],
                'transactions_analyzed': result['metadata']['transactions_analyzed'],
                'analysis_start_date': None,  # Would be populated from actual analysis
                'analysis_end_date': None,
                'model_version': result['metadata']['model_version'],
                'generated_at': result['metadata']['generated_at']
            },
            'error_message': None
        }
    
    def _error_response(self, status: str, message: str) -> Dict:
        return {
            'status': status,
            'predictions': [],
            'metadata': None,
            'error_message': message
        }
    
    def _feedback_error(self, status: str, message: str) -> Dict:
        return {
            'status': status,
            'processed_count': 0,
            'failed_items': [],
            'message': message,
            'retraining_triggered': False
        }


#  gRPC Server Setup (for standalone testing)
def serve_grpc(port: int = 50051):
    # This would use the generated pb2_grpc classes
    # For now, this is a placeholder showing the pattern
    print(f"gRPC server would start on port {port}")
    print("In Azure Functions, use HTTP triggers with JSON payloads instead")
    

if __name__ == '__main__':
    # Quick test of the servicer
    servicer = PredictiveCalendarServicer()
