from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def load_dataset(data_path=DATA_PATH):
    """Load and validate the credit card fraud dataset."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    return df


def inspect_dataset(df):
    """Display basic information about the dataset."""

    if df.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    print("\nDataset Information")
    print("-" * 50)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nClass Distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nFraud Percentage:")
    print(
        round(
            df[TARGET_COLUMN].mean() * 100,
            4
        ),
        "%"
    )


def clean_dataset(df, drop_duplicates=False):
    """
    Validate the dataset and optionally remove exact duplicates.

    Duplicate removal is configurable because identical transactions
    are not automatically proof of erroneous records.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column is missing.")

    if df.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values. "
            "Handle them before training."
        )

    if drop_duplicates:
        duplicate_count = df.duplicated().sum()

        print(f"Removing {duplicate_count} duplicate rows.")

        df = df.drop_duplicates().reset_index(drop=True)

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError(
            "Both legitimate and fraudulent classes are required."
        )

    feature_columns = [
        column for column in df.columns
        if column != TARGET_COLUMN
    ]

    if not all(
        pd.api.types.is_numeric_dtype(df[column])
        for column in feature_columns
    ):
        raise ValueError("All model features must be numeric.")

    feature_values = df[feature_columns].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Features contain infinite or non-finite values."
        )

    return df


def split_dataset(df):
    """Create stratified 60/20/20 train, validation and test splits."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two records for stratified splitting."
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def scale_features(X_train, X_val, X_test):
    """
    Fit the scaler only on training data to prevent data leakage.
    Return DataFrames with their original feature names and order.
    """

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


def prepare_data(drop_duplicates=False):
    """
    Prepare the complete dataset for model training.

    Returns raw splits, scaled splits, fitted scaler and feature names.
    """

    df = load_dataset()

    inspect_dataset(df)

    df = clean_dataset(
        df,
        drop_duplicates=drop_duplicates
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(df)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    feature_names = list(X_train.columns)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "scaler": scaler,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    prepared_data = prepare_data()

    print("\nPreprocessing completed.")
    print(
        "Training samples:",
        len(prepared_data["X_train"])
    )
    print(
        "Validation samples:",
        len(prepared_data["X_val"])
    )
    print(
        "Test samples:",
        len(prepared_data["X_test"])
    )