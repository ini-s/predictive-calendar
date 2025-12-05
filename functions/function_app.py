import azure.functions as func
import json
import logging
import sys
import os
from datetime import datetime

#Add paths for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))

#Import our modules
from models.prediction_model import PredictiveCalendarService
from utils.grpc_service import PredictiveCalendarServicer

#Initialize the Azure Functions app
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

#Initialize the prediction service (singleton for the function app lifecycle)
prediction_service = PredictiveCalendarService()
grpc_servicer = PredictiveCalendarServicer(prediction_service)


#ENDPOINT 1: Generate Predictions
@app.function_name(name="GeneratePredictions")
@app.route(route="predictions", methods=["POST"])
async def generate_predictions(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Generate Predictions endpoint called')
    
    try:
        # Parse request body
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "predictions": [],
                "metadata": None,
                "error_message": "Invalid JSON in request body"
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    #Validate required fields
    if not req_body.get('user_id'):
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "predictions": [],
                "metadata": None,
                "error_message": "user_id is required"
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    if not req_body.get('target_month'):
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "predictions": [],
                "metadata": None,
                "error_message": "target_month is required (format: YYYY-MM)"
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    #Process the prediction request through the gRPC servicer
    #This ensures the same logic whether called via HTTP or gRPC
    result = grpc_servicer.GeneratePredictions(req_body)
    
    #Determine HTTP status code based on response status
    status_code = 200
    if result['status'] in ['INVALID_REQUEST', 'USER_NOT_FOUND']:
        status_code = 400
    elif result['status'] == 'INSUFFICIENT_DATA':
        status_code = 422  # Unprocessable Entity
    elif result['status'] == 'FAILURE':
        status_code = 500
    
    return func.HttpResponse(
        json.dumps(result, default=str),
        status_code=status_code,
        mimetype="application/json"
    )


# ENDPOINT 2: Submit Feedback
@app.function_name(name="SubmitFeedback")
@app.route(route="feedback", methods=["POST"])
async def submit_feedback(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Submit Feedback endpoint called')
    
    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "processed_count": 0,
                "failed_items": [],
                "message": "Invalid JSON in request body",
                "retraining_triggered": False
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    #Validate required fields
    if not req_body.get('user_id'):
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "processed_count": 0,
                "failed_items": [],
                "message": "user_id is required",
                "retraining_triggered": False
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    if not req_body.get('feedback_items'):
        return func.HttpResponse(
            json.dumps({
                "status": "INVALID_REQUEST",
                "processed_count": 0,
                "failed_items": [],
                "message": "feedback_items is required",
                "retraining_triggered": False
            }),
            status_code=400,
            mimetype="application/json"
        )
    
    #Process feedback through the gRPC servicer
    result = grpc_servicer.SubmitFeedback(req_body)
    
    #Determine HTTP status code
    status_code = 200
    if result['status'] == 'INVALID_REQUEST':
        status_code = 400
    elif result['status'] == 'FAILURE':
        status_code = 500
    
    return func.HttpResponse(
        json.dumps(result, default=str),
        status_code=status_code,
        mimetype="application/json"
    )


#gRPC-Web Compatibility Endpoint
@app.function_name(name="GrpcWebProxy")
@app.route(route="grpc/{method}", methods=["POST"])
async def grpc_web_proxy(req: func.HttpRequest) -> func.HttpResponse:
    method = req.route_params.get('method')
    logging.info(f'gRPC-Web proxy called for method: {method}')
    
    if method not in ['GeneratePredictions', 'SubmitFeedback']:
        return func.HttpResponse(
            json.dumps({
                "error": f"Unknown method: {method}",
                "available_methods": [
                    "GeneratePredictions",
                    "SubmitFeedback"
                ]
            }),
            status_code=404,
            mimetype="application/json"
        )
    
    try:
        req_body = req.get_json()
    except ValueError:
        req_body = {}
    
    #Dispatch to appropriate handler
    if method == 'GeneratePredictions':
        result = grpc_servicer.GeneratePredictions(req_body)
    elif method == 'SubmitFeedback':
        result = grpc_servicer.SubmitFeedback(req_body)
    
    return func.HttpResponse(
        json.dumps(result, default=str),
        status_code=200,
        mimetype="application/json"
    )