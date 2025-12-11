# ALAT Predictive Calendar - Serverless Prediction System

## Project Overview

The ALAT Predictive Calendar is a serverless, machine learning-powered
system that analyzes financial transaction history to predict future
recurring expenses. Built on Azure Functions with a clean, modular
architecture, this system detects patterns in spending habits and
generates a calendar of expected future transactions, helping users
anticipate bills, subscriptions, loan payments, and other regular
financial obligations.

The system employs rule-based pattern detection enhanced by machine
learning clustering to identify recurring transactions. Its unique
feedback learning mechanism allows it to adapt and improve predictions
based on user interactions, creating a personalized prediction engine
that evolves with user behavior.

## Project Structure

    predictive_calendar/
    ├── protos/
    │   └── predictive_calendar.proto
    ├── models/
    │   └── prediction_model.py
    ├── utils/
    │   └── grpc_service.py
    ├── functions/
    │   └── function_app.py
    ├── tests/
    │   └── test_prediction_model.py
    ├── data/
    │   └── user_transactions_mapping.csv
    ├── requirements.txt
    ├── host.json
    ├── local.settings.json
    └── README.md

## System Architecture

  
                Azure Functions (functions/)                
        HTTP Endpoints → Request Validation → Response       

                               ↓
   
            gRPC Service Layer (utils/grpc_service.py)    
         Unified request handling for HTTP/gRPC protocols 

                                ↓

      Core Prediction Engine (models/prediction_model.py) 
       Pattern Detection → Prediction → Feedback Learning 
   
                                ↓
   
                 Data Layer (protos/, data/)              
       Contract Definitions → Test Data → Configuration   
    

## Key Components

### Azure Functions Entry Point (`functions/function_app.py`)

-   HTTP endpoints for predictions, feedback, and gRPC-Web
    compatibility\
-   Request validation and error handling\
-   Singleton service instances for efficient resource usage

### Core Prediction Engine (`models/prediction_model.py`)

-   **RuleBasedPatternDetector**
-   **PredictionGenerator**
-   **FeedbackLearner**
-   **PredictiveCalendarService**

### gRPC Service Layer (`utils/grpc_service.py`)

-   Unified HTTP and gRPC request handling\
-   Request transformation and response formatting

### Protocol Definitions (`protos/predictive_calendar.proto`)

-   Defines gRPC contract\
-   Ensures consistent data structures across services

## How It Works

### Pattern Detection Methods

-   Receiver-based grouping\
-   Type & amount clustering\
-   Weekly recurrence detection

### Prediction Generation

-   Next occurrence estimation\
-   Amount estimation\
-   Confidence scoring\
-   Reasoning explanation

### Continuous Learning

-   Adjusts category weights\
-   Filters repeated discards\
-   Triggers retraining\
-   Persists learning

## Local Development and Testing

### Prerequisites

-   Python 3.8+\
-   Azure Functions Core Tools\
-   Virtual environment tool

### Installation

``` bash
git clone https://github.com/ini-s/predictive-calendar
cd predictive_calendar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Local Settings

``` json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}
```

### Run Azure Functions

``` bash
cd functions
func start
```

Endpoints: - `/api/predictions` - `/api/feedback` - `/api/grpc/{method}`

### Testing with curl

Prediction and feedback curl commands included in original text.

## Running Tests

``` bash
pytest
pytest -v
pytest tests/test_prediction_model.py
```


## Configuration and Customization

-   Transaction categories\
-   Pattern detection tuning\
-   gRPC regeneration commands\
-   Azure Functions configuration

## Deployment to Azure

``` bash
pip install --target="./.python_packages/lib/site-packages" -r requirements.txt
func azure functionapp publish <YourFunctionAppName>
```

## Extending the System

-   Add pattern types\
-   Add transaction categories\
-   Extend protocol buffers\
-   Update service layer

## Future Enhancements

-   Real-time data ingestion\
-   Advanced pattern logic\
-   Calendar export\
-   Spending analytics
