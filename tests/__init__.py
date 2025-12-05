"""
ALAT Predictive Calendar - Tests Package

This package contains comprehensive tests for the prediction model,
pattern detection, feedback learning, and Azure Functions endpoints.

Run all tests with: pytest tests/ -v
"""

import sys
import os

# Add parent directory to path for imports during testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .test_prediction_model import (
    TestRuleBasedPatternDetector,
    TestPredictionGenerator,
    TestFeedbackLearner,
    TestPredictiveCalendarService,
    TestPredictionAccuracy
)

__all__ = [
    'TestRuleBasedPatternDetector',
    'TestPredictionGenerator', 
    'TestFeedbackLearner',
    'TestPredictiveCalendarService',
    'TestPredictionAccuracy'
]
