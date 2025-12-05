import sys
import os

#Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Import the main function app and endpoints
from .function_app import (
    app,
    generate_predictions,
    submit_feedback,
    grpc_web_proxy
)

__all__ = [
    'app',
    'generate_predictions',
    'submit_feedback',
    'grpc_web_proxy'
]