from fastapi.testclient import TestClient

from app.main import app
from app.services import prediction


client = TestClient(app)


def create_valid_transaction():
    """Create a sample transaction with all 30 required features."""

    transaction = {
        "Time": 10000.0,
        "Amount": 150.0
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.1

    return transaction


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint(monkeypatch):
    """Test prediction endpoint using a mocked prediction service."""

    def mock_predict(transaction_data):
        return {
            "transaction_id": "test-123",
            "fraud_probability": 0.95,
            "is_fraud": True,
            "decision": "FRAUD",
            "model_name": "Random Forest",
            "threshold": 0.5
        }

    monkeypatch.setattr(
        "app.main.predict_transaction",
        mock_predict
    )

    response = client.post(
        "/predict",
        json=create_valid_transaction()
    )

    assert response.status_code == 200

    result = response.json()

    assert result["is_fraud"] is True
    assert result["decision"] == "FRAUD"
    assert result["fraud_probability"] == 0.95


def test_missing_feature():
    transaction = create_valid_transaction()
    transaction.pop("V1")

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_extra_feature():
    transaction = create_valid_transaction()
    transaction["UnknownFeature"] = 100

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


def test_invalid_negative_amount():
    transaction = create_valid_transaction()
    transaction["Amount"] = -100

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422