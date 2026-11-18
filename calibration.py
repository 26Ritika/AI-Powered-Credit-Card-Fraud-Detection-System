from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score
)


def calibrate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    method="sigmoid",
    cv=5
):
    """
    Calibrate model probabilities using cross-validation.
    """

    if method not in ("sigmoid", "isotonic"):
        raise ValueError(
            "Method must be either 'sigmoid' or 'isotonic'."
        )

    if cv < 2:
        raise ValueError("cv must be at least 2.")

    if y_train.value_counts().min() < cv:
        raise ValueError(
            "Each class must have at least cv training samples."
        )

    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=cv
        )
    except TypeError:
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model,
            method=method,
            cv=cv
        )

    calibrated_model.fit(X_train, y_train)

    val_probabilities = calibrated_model.predict_proba(X_val)[:, 1]

    brier_score = brier_score_loss(
        y_val,
        val_probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    print("\n========== CALIBRATION RESULTS ==========")
    print(f"Calibration method: {method}")
    print(f"Brier Score: {brier_score:.6f}")
    print(f"Validation PR-AUC: {pr_auc:.6f}")

    return {
        "calibrated_model": calibrated_model,
        "validation_probabilities": val_probabilities,
        "brier_score": float(brier_score),
        "pr_auc": float(pr_auc)
    }