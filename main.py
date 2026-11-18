from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import TransactionInput, PredictionResponse
from app.services.prediction import predict_transaction


app = FastAPI(
    title="AI-Powered Credit Card Fraud Detection API",
    description=(
        "Machine learning API for detecting potentially "
        "fraudulent credit card transactions."
    ),
    version="1.0.0"
)


@app.get("/", tags=["General"])
def home():
    return {
        "message": "AI Credit Card Fraud Detection API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "fraud-detection-api"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(transaction: TransactionInput):

    try:
        transaction_data = transaction.model_dump()

        result = predict_transaction(transaction_data)

        return PredictionResponse(**result)

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An internal prediction error occurred."
        )