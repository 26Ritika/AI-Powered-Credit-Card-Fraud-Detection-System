import numpy as np
import pytest

from src.preprocessing import prepare_data


@pytest.fixture(scope="module")
def processed_data():
    """Prepare the dataset once for all tests in this module."""
    return prepare_data()


def test_data_splits_are_not_empty(processed_data):
    data = processed_data

    assert len(data["X_train"]) > 0
    assert len(data["X_val"]) > 0
    assert len(data["X_test"]) > 0


def test_feature_and_target_lengths_match(processed_data):
    data = processed_data

    assert len(data["X_train"]) == len(data["y_train"])
    assert len(data["X_val"]) == len(data["y_val"])
    assert len(data["X_test"]) == len(data["y_test"])


def test_all_splits_contain_both_classes(processed_data):
    data = processed_data

    for target in ("y_train", "y_val", "y_test"):
        assert set(data[target].unique()) == {0, 1}


def test_feature_columns_are_consistent(processed_data):
    data = processed_data

    expected = list(data["feature_names"])

    assert list(data["X_train"].columns) == expected
    assert list(data["X_val"].columns) == expected
    assert list(data["X_test"].columns) == expected


def test_scaled_data_shapes_match(processed_data):
    data = processed_data

    assert data["X_train_scaled"].shape == data["X_train"].shape
    assert data["X_val_scaled"].shape == data["X_val"].shape
    assert data["X_test_scaled"].shape == data["X_test"].shape


def test_scaled_data_contains_finite_values(processed_data):
    data = processed_data

    for key in (
        "X_train_scaled",
        "X_val_scaled",
        "X_test_scaled"
    ):
        assert np.isfinite(data[key]).all().all()


def test_scaler_is_fitted(processed_data):
    data = processed_data

    assert hasattr(data["scaler"], "mean_")
    assert hasattr(data["scaler"], "scale_")