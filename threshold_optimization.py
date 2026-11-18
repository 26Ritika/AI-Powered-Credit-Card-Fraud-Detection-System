import numpy as np

from sklearn.metrics import confusion_matrix


def optimize_threshold(
    y_true,
    probabilities,
    false_negative_cost=5000,
    false_positive_cost=200
):
    """
    Find the classification threshold with the lowest
    estimated business cost.

    Default costs are illustrative and must be replaced
    with business-approved values for real deployment.
    """

    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("Business costs cannot be negative.")

    if false_negative_cost == 0 and false_positive_cost == 0:
        raise ValueError("At least one business cost must be positive.")

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Input data cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = None
    minimum_cost = float("inf")
    best_details = None

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        total_cost = (
            fn * false_negative_cost
            + fp * false_positive_cost
        )

        if total_cost < minimum_cost:
            minimum_cost = total_cost
            best_threshold = float(threshold)

            best_details = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_cost": float(total_cost)
            }

    print("\n========== THRESHOLD OPTIMIZATION ==========")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Estimated Business Cost: {minimum_cost:.2f}")
    print(f"False Positives: {best_details['false_positives']}")
    print(f"False Negatives: {best_details['false_negatives']}")

    return {
        "best_threshold": best_threshold,
        "minimum_cost": float(minimum_cost),
        "details": best_details
    }