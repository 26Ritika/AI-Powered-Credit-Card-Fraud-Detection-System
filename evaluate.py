import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classification model on test data.
    """

    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = model.predict_proba(X_test)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Model generated invalid prediction probabilities.")

    predictions = (probabilities >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, probabilities)
    roc_auc = roc_auc_score(y_test, probabilities)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_test, predictions, beta=2, zero_division=0
    )

    matrix = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }

    print("\n========== TEST RESULTS ==========")
    print(f"PR-AUC:    {pr_auc:.5f}")
    print(f"ROC-AUC:   {roc_auc:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall:    {recall:.5f}")
    print(f"F1-score:  {f1:.5f}")
    print(f"F2-score:  {f2:.5f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    return metrics