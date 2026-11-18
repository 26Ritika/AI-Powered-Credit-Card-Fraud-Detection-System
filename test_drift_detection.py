import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detection import (
    calculate_psi,
    detect_data_drift
)


def test_identical_distributions_have_zero_psi():
    data = np.random.default_rng(42).normal(0, 1, 1000)

    psi = calculate_psi(
        reference=data,
        current=data
    )

    assert psi == pytest.approx(0.0)


def test_shifted_distribution_detects_drift():
    rng = np.random.default_rng(42)

    reference = rng.normal(0, 1, 5000)
    current = rng.normal(2, 1, 5000)

    psi = calculate_psi(reference, current)

    assert psi > 0.2


def test_empty_reference_data_raises_error():
    with pytest.raises(ValueError):
        calculate_psi([], [1, 2, 3])


def test_empty_current_data_raises_error():
    with pytest.raises(ValueError):
        calculate_psi([1, 2, 3], [])


def test_invalid_bin_count_raises_error():
    with pytest.raises(ValueError):
        calculate_psi([1, 2, 3], [1, 2, 3], bins=1)


def test_detect_data_drift_returns_report():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "Amount": rng.normal(100, 10, 1000),
        "Time": rng.normal(500, 50, 1000)
    })

    current = reference.copy()

    report = detect_data_drift(reference, current)

    assert isinstance(report, pd.DataFrame)
    assert set(report.columns) == {"feature", "psi", "status"}
    assert len(report) == 2


def test_mismatched_columns_raise_error():
    reference = pd.DataFrame({
        "Amount": [100, 200, 300]
    })

    current = pd.DataFrame({
        "Time": [10, 20, 30]
    })

    with pytest.raises(ValueError):
        detect_data_drift(reference, current)


def test_empty_dataset_raises_error():
    reference = pd.DataFrame({"Amount": []})
    current = pd.DataFrame({"Amount": []})

    with pytest.raises(ValueError):
        detect_data_drift(reference, current)


def test_invalid_threshold_raises_error():
    reference = pd.DataFrame({"Amount": [100, 200, 300]})
    current = pd.DataFrame({"Amount": [100, 200, 300]})

    with pytest.raises(ValueError):
        detect_data_drift(reference, current, threshold=0)


