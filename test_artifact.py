import joblib
import pytest

from sklearn.linear_model import LogisticRegression

from app.services import prediction


@pytest.fixture
def valid_artifact(tmp_path, monkeypatch):
    """Create a temporary valid model artifact."""

    model = LogisticRegression()

    model.fit(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [0, 0, 1, 1]
    )

    artifact = {
        "model": model,
        "scaler": None,
        "threshold": 0.5,
        "model_name": "Logistic Regression",
        "feature_names": ["feature_1", "feature_2"]
    }

    artifact_path = tmp_path / "test_model.joblib"

    joblib.dump(artifact, artifact_path)

    monkeypatch.setattr(
        prediction,
        "MODEL_PATH",
        artifact_path
    )

    prediction.load_model_artifact.cache_clear()

    yield artifact

    prediction.load_model_artifact.cache_clear()


def test_load_valid_artifact(valid_artifact):
    loaded = prediction.load_model_artifact()

    assert loaded["model_name"] == "Logistic Regression"
    assert loaded["threshold"] == 0.5
    assert len(loaded["feature_names"]) == 2


def test_missing_artifact(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing_model.joblib"

    monkeypatch.setattr(
        prediction,
        "MODEL_PATH",
        missing_path
    )

    prediction.load_model_artifact.cache_clear()

    with pytest.raises(FileNotFoundError):
        prediction.load_model_artifact()

    prediction.load_model_artifact.cache_clear()


def test_invalid_artifact(tmp_path, monkeypatch):
    invalid_path = tmp_path / "invalid_model.joblib"

    joblib.dump(
        {"model": "invalid"},
        invalid_path
    )

    monkeypatch.setattr(
        prediction,
        "MODEL_PATH",
        invalid_path
    )

    prediction.load_model_artifact.cache_clear()

    with pytest.raises(ValueError):
        prediction.load_model_artifact()

    prediction.load_model_artifact.cache_clear()


def test_prediction_with_valid_artifact(valid_artifact):
    transaction = {
        "feature_1": 0.5,
        "feature_2": 0.5
    }

    result = prediction.predict_transaction(transaction)

    assert 0 <= result["fraud_probability"] <= 1
    assert isinstance(result["is_fraud"], bool)
    assert result["decision"] in ["FRAUD", "LEGITIMATE"]
    assert result["model_name"] == "Logistic Regression"