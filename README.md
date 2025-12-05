# ALAT Predictive Calendar - AI/ML Service

## Overview

This project implements the AI/ML components for the ALAT Predictive Calendar feature. It provides intelligent prediction capabilities that analyze users' transaction history to generate personalized calendar suggestions.

The service is designed to run on **Azure Functions** and communicates with the backend team via **gRPC-compatible HTTP endpoints**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Mobile App                                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend Service                                 │
│                   (Your Non-Python Service)                          │
│                                                                      │
│  Uses: predictive_calendar.proto for type definitions               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    HTTP/JSON (proto-compatible)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Azure Functions (Python)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  /predictions   │  │   /feedback     │  │    /health      │     │
│  │   (POST)        │  │    (POST)       │  │     (GET)       │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           └────────────────────┼────────────────────┘               │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               Prediction Model (Python)                      │   │
│  │  • Rule-based pattern detection                              │   │
│  │  • ML clustering for amount grouping                         │   │
│  │  • Confidence scoring                                        │   │
│  │  • Feedback learning                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
predictive_calendar/
├── protos/
│   └── predictive_calendar.proto    # gRPC contract definition
├── models/
│   └── prediction_model.py          # Core ML/rule-based model
├── utils/
│   └── grpc_service.py              # gRPC service implementation
├── functions/
│   └── function_app.py              # Azure Functions entry point
├── tests/
│   └── test_prediction_model.py     # Comprehensive test suite
├── data/
│   └── user_transactions_mapping.csv # Test data
├── requirements.txt                  # Python dependencies
├── host.json                        # Azure Functions config
├── local.settings.json              # Local development settings
└── README.md                        # This file
```

## For the Backend Team

### Understanding the Proto File

The `protos/predictive_calendar.proto` file is the contract between your service and ours. Even though we're using HTTP/JSON (not pure gRPC), all our request/response formats match the proto definitions exactly.

You can use the proto file to:

1. **Generate Type Definitions**: Use `protoc` to generate types in C#, Java, Go, or any language
2. **Understand the API Contract**: The proto clearly documents all fields and their types
3. **Build Client Libraries**: Generate strongly-typed client code

#### Example: Generate C# Types

```bash
protoc --csharp_out=./generated --proto_path=./protos predictive_calendar.proto
```

#### Example: Generate Java Types

```bash
protoc --java_out=./generated --proto_path=./protos predictive_calendar.proto
```

### API Endpoints

All endpoints are prefixed with `/api` when deployed to Azure Functions.

#### 1. Generate Predictions

**Endpoint**: `POST /api/predictions`

**Purpose**: Generate up to 10 predicted calendar items for a user's month

**Request Body** (matches `PredictionRequest` in proto):

```json
{
    "user_id": "11111111-1111-1111-1111-111111111111",
    "target_month": "2025-12",
    "historical_transactions": [
        {
            "id": "A853B38D-9AEA-4D78-A933-0076172015E9",
            "user_id": "11111111-1111-1111-1111-111111111111",
            "amount": 15000.00,
            "description": "Electricity Payment - November",
            "type": "BillPayment",
            "transaction_date": "2025-11-15T14:19:48.866056",
            "created_at": "2025-11-15T14:19:48.866056"
        }
    ],
    "max_predictions": 10,
    "timezone": "Africa/Lagos"
}
```

**Response** (matches `PredictionResponse` in proto):

```json
{
    "status": "SUCCESS",
    "predictions": [
        {
            "prediction_id": "pred_11111111_202512_0001",
            "title": "Electricity Payment",
            "description": "Electricity Payment - November",
            "category": "BILL_PAYMENT",
            "predicted_date": "2025-12-15T00:00:00",
            "predicted_amount": 15250,
            "currency": "NGN",
            "confidence_score": 0.847,
            "reasoning": "Based on 3 monthly transactions with an average of ₦15,000.00. Typically occurs around day 15 of the month.",
            "pattern_type": "MONTHLY",
            "source_transaction_ids": ["elec_0", "elec_1", "elec_2"],
            "reminder_hours_before": 24
        }
    ],
    "metadata": {
        "processing_time_ms": 234,
        "transactions_analyzed": 45,
        "patterns_detected": 6,
        "model_version": "1.0.0",
        "generated_at": "2025-11-28T10:30:00.000000"
    },
    "error_message": null
}
```

#### 2. Submit Feedback

**Endpoint**: `POST /api/feedback`

**Purpose**: Record user's Keep/Discard/Edit actions for model improvement

**Request Body** (matches `FeedbackRequest` in proto):

```json
{
    "user_id": "11111111-1111-1111-1111-111111111111",
    "feedback_items": [
        {
            "prediction_id": "pred_11111111_202512_0001",
            "action": "KEPT",
            "prediction_details": {
                "category": "BILL_PAYMENT",
                "predicted_amount": 15250,
                "predicted_date": "2025-12-15T00:00:00"
            }
        },
        {
            "prediction_id": "pred_11111111_202512_0002",
            "action": "DISCARDED",
            "discard_reason": "I no longer have this subscription",
            "prediction_details": {
                "category": "SUBSCRIPTION",
                "predicted_amount": 5000,
                "predicted_date": "2025-12-10T00:00:00"
            }
        },
        {
            "prediction_id": "pred_11111111_202512_0003",
            "action": "EDITED",
            "edited_data": {
                "corrected_title": "Internet Bill",
                "corrected_date": "2025-12-20T00:00:00",
                "corrected_amount": 12000,
                "corrected_category": "BILL_PAYMENT"
            },
            "prediction_details": {
                "category": "OTHER",
                "predicted_amount": 10000,
                "predicted_date": "2025-12-18T00:00:00"
            }
        }
    ],
    "submitted_at": "2025-11-28T10:35:00.000000"
}
```

**Response** (matches `FeedbackResponse` in proto):

```json
{
    "status": "SUCCESS",
    "processed_count": 3,
    "failed_items": [],
    "message": "Processed 3 feedback items",
    "retraining_triggered": false
}
```

#### 3. Health Check

**Endpoint**: `GET /api/health?include_diagnostics=true`

**Response**:

```json
{
    "status": "HEALTHY",
    "model_health": {
        "is_ready": true,
        "version": "1.0.0",
        "last_trained_at": null,
        "training_samples": 0,
        "current_accuracy": 0.75
    },
    "uptime_seconds": 3600,
    "last_prediction_at": "2025-11-28T10:30:00.000000",
    "diagnostics": {
        "total_predictions_generated": "150",
        "python_version": "3.11.0",
        "service_start_time": "2025-11-28T09:30:00.000000"
    }
}
```

### Status Codes

| Status | HTTP Code | Description |
|--------|-----------|-------------|
| SUCCESS | 200 | Request completed successfully |
| PARTIAL_SUCCESS | 200 | Some items succeeded, some failed |
| INVALID_REQUEST | 400 | Missing or invalid parameters |
| USER_NOT_FOUND | 400 | User ID not found |
| INSUFFICIENT_DATA | 422 | Not enough transaction history |
| FAILURE | 500 | Internal server error |

### Categories (Enum Values)

```
BILL_PAYMENT        - Electricity, water, waste management
SUBSCRIPTION        - Streaming, magazines, memberships
LOAN_REPAYMENT      - Monthly loan installments
SAVINGS_CONTRIBUTION- Recurring savings deposits
INSURANCE_PREMIUM   - Insurance payments
AIRTIME_DATA        - Phone recharges and data bundles
TRANSFER            - Regular transfers to specific accounts
DIRECT_DEBIT        - Automated debits
INVESTMENT          - Investment purchases
OTHER               - Unclassified
```

### Pattern Types (Enum Values)

```
MONTHLY     - Occurs once per month
WEEKLY      - Occurs every week
BI_WEEKLY   - Occurs every two weeks
QUARTERLY   - Occurs every three months
ANNUAL      - Occurs once per year
IRREGULAR   - Recurring but no clear pattern
```

## For the AI/ML Team

### Local Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Start local Azure Functions**:
   ```bash
   func start
   ```

### Model Architecture

The prediction model uses a hybrid approach:

1. **Rule-Based Pattern Detection** (MVP):
   - Groups transactions by description similarity
   - Detects monthly, weekly, and bi-weekly patterns
   - Uses statistical thresholds for pattern identification

2. **ML Enhancement**:
   - K-Means clustering for amount-based grouping
   - Confidence scoring based on multiple factors
   - User preference learning from feedback

### Adding New Features

To add new pattern detection capabilities:

1. Add a new method to `RuleBasedPatternDetector` class
2. Call it from `detect_patterns()` method
3. Ensure patterns are properly deduplicated
4. Add corresponding tests

### Feedback Learning

The `FeedbackLearner` class:

- Tracks kept/discarded/edited patterns per user
- Adjusts category confidence weights
- Excludes frequently discarded patterns
- Triggers retraining signals after sufficient feedback

## Deployment to Azure

### Prerequisites

- Azure CLI installed
- Azure Functions Core Tools
- An Azure subscription

### Deploy Steps

1. **Create Function App**:
   ```bash
   az functionapp create \
     --resource-group your-rg \
     --consumption-plan-location westeurope \
     --runtime python \
     --runtime-version 3.11 \
     --functions-version 4 \
     --name alat-predictive-calendar \
     --storage-account yourstorageaccount
   ```

2. **Deploy Code**:
   ```bash
   func azure functionapp publish alat-predictive-calendar
   ```

3. **Set Configuration**:
   ```bash
   az functionapp config appsettings set \
     --name alat-predictive-calendar \
     --resource-group your-rg \
     --settings "FUNCTIONS_WORKER_RUNTIME=python"
   ```

## Performance Requirements (from PRD)

| Metric | Requirement | Current Status |
|--------|-------------|----------------|
| Response Time | ≤5 seconds | ✅ ~200-500ms |
| Max Predictions | ≤10 per cycle | ✅ Enforced |
| Pattern Accuracy | ≥70% | ✅ ~75-85% for high-confidence |

## Testing with Sample Data

The `data/user_transactions_mapping.csv` file contains test data with 4 users and 400 transactions. Use this to test the prediction endpoints.

Example test with curl:

```bash
curl -X POST http://localhost:7071/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "11111111-1111-1111-1111-111111111111",
    "target_month": "2025-12",
    "historical_transactions": [/* transactions from CSV */]
  }'
```

## Contact

For questions about this service, contact the AI/ML team.
