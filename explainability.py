from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap


def generate_shap_explanations(
    model,
    X_background,
    X_sample,
    model_name,
    output_dir="models/shap"
):
    """
    Generate global and individual SHAP explanations.

    Important:
    - For Logistic Regression, provide scaled features.
    - For Random Forest and XGBoost, provide raw features.
    - Background and sample data must use the same feature format
      used to train the selected model.
    """

    if not isinstance(X_background, pd.DataFrame):
        X_background = pd.DataFrame(X_background)

    if not isinstance(X_sample, pd.DataFrame):
        X_sample = pd.DataFrame(
            X_sample,
            columns=X_background.columns
        )

    if X_background.empty or X_sample.empty:
        raise ValueError("Background and sample data cannot be empty.")

    if list(X_background.columns) != list(X_sample.columns):
        raise ValueError("Background and sample features must match.")

    if model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(
            model,
            X_background
        )
    elif model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    shap_values = explainer.shap_values(X_sample)

    # Select the fraud class for binary classification.
    if isinstance(shap_values, list):
        values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    base_values = explainer.expected_value

    if hasattr(base_values, "__len__"):
        base_values = base_values[1]

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.to_numpy(),
        feature_names=list(X_sample.columns)
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Global feature importance
    plt.figure()
    shap.plots.bar(
        explanation,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Global feature impact
    plt.figure()
    shap.plots.beeswarm(
        explanation,
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Individual transaction explanation
    plt.figure()
    shap.plots.waterfall(
        explanation[0],
        max_display=15,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_path / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nSHAP explanations generated successfully.")
    print(f"Saved at: {output_path.resolve()}")

    return explanation