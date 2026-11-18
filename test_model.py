import numpy as np
import pytest

from src.preprocessing import prepare_data
from src.train import train_models


@pytest.fixture(scope="module")
def trained_pipeline():
    """
    Prepare data and train all candidate models once
    for this test module.
    """

    data = prepare_data()
    results = train_models(data)

    return data, results


def test_models_are_trained(trained_pipeline):
    data, results = trained_pipeline

    assert len(results["models"]) == 3

    assert "Logistic Regression" in results["models"]
    assert "Random Forest" in results["models"]
    assert "XGBoost" in results["models"]


def test_best_model_is_selected(trained_pipeline):
    data, results = trained_pipeline

    assert results["best_model"] is not None

    assert (
        results["best_model_name"]
        in results["models"]
    )


def test_validation_pr_auc_is_valid(trained_pipeline):
    data, results = trained_pipeline

    for model_name, score in results["results"].items():

        assert 0 <= score <= 1, (
            f"Invalid PR-AUC for {model_name}"
        )


def test_predictions_are_valid_probabilities(trained_pipeline):
    data, results = trained_pipeline

    best_model = results["best_model"]
    best_model_name = results["best_model_name"]

    if best_model_name == "Logistic Regression":
        X_test = data["X_test_scaled"]
    else:
        X_test = data["X_test"]

    probabilities = best_model.predict_proba(X_test)[:, 1]

    assert len(probabilities) == len(data["y_test"])

    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_training_and_test_data_contain_both_classes(trained_pipeline):
    data, results = trained_pipeline

    assert set(data["y_train"].unique()) == {0, 1}
    assert set(data["y_val"].unique()) == {0, 1}
    assert set(data["y_test"].unique()) == {0, 1}