from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def train_models(data):
    """
    Train Logistic Regression, Random Forest and XGBoost.
    Select the best model using validation PR-AUC.
    """

    X_train = data["X_train"]
    X_val = data["X_val"]

    X_train_scaled = data["X_train_scaled"]
    X_val_scaled = data["X_val_scaled"]

    y_train = data["y_train"]
    y_val = data["y_val"]

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    }

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError("Training data contains no fraud cases.")

    scale_pos_weight = negative_count / positive_count

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    results = {}
    trained_models = {}

    for name, model in models.items():

        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            probabilities = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_val)[:, 1]

        pr_auc = average_precision_score(y_val, probabilities)

        trained_models[name] = model
        results[name] = float(pr_auc)

        print(f"{name} Validation PR-AUC: {pr_auc:.5f}")

    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]

    print("\nBest Model:", best_model_name)
    print("Best Validation PR-AUC:", results[best_model_name])

    return {
        "models": trained_models,
        "results": results,
        "best_model": best_model,
        "best_model_name": best_model_name
    }


if __name__ == "__main__":
    from src.preprocessing import prepare_data

    dataset = prepare_data()
    training_results = train_models(dataset)