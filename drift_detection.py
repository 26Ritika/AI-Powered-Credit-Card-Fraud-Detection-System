import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):
    """
    Calculate Population Stability Index (PSI)
    between reference and current numerical data.
    """

    if bins < 2:
        raise ValueError("Number of bins must be at least 2.")

    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Reference and current data cannot be empty.")

    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("Data must contain finite numerical values.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    if len(edges) < 2:
        # A constant reference feature cannot define useful bins.
        # Any change from that constant is reported as drift.
        return 0.0 if np.all(current == reference[0]) else float("inf")

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    epsilon = 1e-6

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def detect_data_drift(
    reference_data,
    current_data,
    threshold=0.2,
    bins=10
):
    """
    Detect numerical feature drift using PSI.

    PSI interpretation:
    Below 0.10: No significant drift
    0.10 to 0.20: Moderate drift
    Above 0.20: Significant drift
    """

    if reference_data.empty or current_data.empty:
        raise ValueError("Datasets cannot be empty.")

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be a positive finite number.")

    if bins < 2:
        raise ValueError("Number of bins must be at least 2.")

    if set(reference_data.columns) != set(current_data.columns):
        raise ValueError("Reference and current columns must match.")

    current_data = current_data[reference_data.columns]

    results = []

    numeric_columns = reference_data.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        psi_value = calculate_psi(
            reference_data[column],
            current_data[column],
            bins=bins
        )

        if psi_value < 0.1:
            status = "No Significant Drift"
        elif psi_value <= threshold:
            status = "Moderate Drift"
        else:
            status = "Significant Drift"

        results.append({
            "feature": column,
            "psi": psi_value,
            "status": status
        })

    report = pd.DataFrame(results)

    print("\n========== DATA DRIFT REPORT ==========")
    print(report.to_string(index=False))

    return report