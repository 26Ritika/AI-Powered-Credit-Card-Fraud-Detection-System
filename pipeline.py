from pathlib import Path
import joblib

from src.preprocessing import prepare_data
from src.train import train_models
from src.calibration import calibrate_model
from src.threshold_optimization import optimize_threshold
from src.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_detection_model.joblib"


def run_pipeline():
    print("\n========== FRAUD DETECTION PIPELINE ==========\n")

    # 1. Prepare dataset
    print("Step 1: Preparing dataset...")
    data = prepare_data()

    # 2. Train and compare models
    print("\nStep 2: Training models...")
    training_results = train_models(data)

    best_model = training_results["best_model"]
    best_model_name = training_results["best_model_name"]

    # 3. Select correct feature representation
    if best_model_name == "Logistic Regression":
        X_train = data["X_train_scaled"]
        X_val = data["X_val_scaled"]
        X_test = data["X_test_scaled"]
        scaler = data["scaler"]
    else:
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        scaler = None

    # 4. Calibrate probabilities
    print("\nStep 3: Calibrating model...")
    calibration_results = calibrate_model(
        model=best_model,
        X_train=X_train,
        y_train=data["y_train"],
        X_val=X_val,
        y_val=data["y_val"],
        method="sigmoid",
        cv=5
    )

    calibrated_model = calibration_results["calibrated_model"]
    validation_probabilities = calibration_results[
        "validation_probabilities"
    ]

    # 5. Optimize decision threshold
    print("\nStep 4: Optimizing threshold...")
    threshold_results = optimize_threshold(
        y_true=data["y_val"],
        probabilities=validation_probabilities,
        false_negative_cost=5000,
        false_positive_cost=200
    )

    best_threshold = threshold_results["best_threshold"]

    # 6. Evaluate once on untouched test data
    print("\nStep 5: Evaluating final model...")
    test_metrics = evaluate_model(
        model=calibrated_model,
        X_test=X_test,
        y_test=data["y_test"],
        threshold=best_threshold
    )

    # 7. Save model artifact
    print("\nStep 6: Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": calibrated_model,
        "scaler": scaler,
        "threshold": float(best_threshold),
        "model_name": best_model_name,
        "feature_names": data["feature_names"],
        "calibration_method": "sigmoid",
        "validation_brier_score": calibration_results["brier_score"],
        "validation_pr_auc": calibration_results["pr_auc"],
        "test_metrics": test_metrics
    }

    joblib.dump(model_artifact, MODEL_PATH)

    print(f"\nModel saved successfully at: {MODEL_PATH}")
    print("\n========== PIPELINE COMPLETED ==========")

    return {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "model_path": str(MODEL_PATH)
    }


if __name__ == "__main__":
    run_pipeline()