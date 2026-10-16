"""Data loading, feature engineering and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_retention.csv"


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Hillstrom / customer-retention dataset."""
    df = pd.read_csv(path)
    # Fix common typo in original data
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()

    # Binary treatment indicator
    df["treatment"] = (df["offer"] != "No Offer").astype(int)

    # Interactions
    df["history_discount"] = df["history"] * df["used_discount"]
    df["history_bogo"] = df["history"] * df["used_bogo"]
    df["recency_history"] = df["recency"] * df["history"]

    # Log-transform skewed monetary feature
    df["history_log"] = np.log1p(df["history"])

    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature names."""
    categorical = ["zip_code", "channel"]
    # is_referral is already 0/1, treat as numeric
    exclude = {"offer", "conversion", "treatment"}
    numeric = [
        c
        for c in df.columns
        if c not in categorical and c not in exclude
    ]
    return numeric, categorical


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create sklearn ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full preparation pipeline:
    - feature engineering
    - stratified split by treatment + conversion
    - fit preprocessor on train only
    """
    df = create_features(df)

    y = df["conversion"].values
    treatment = df["treatment"].values

    numeric, categorical = get_feature_lists(df)
    feature_cols = numeric + categorical

    X = df[feature_cols]

    # Stratify by treatment × conversion to keep balance
    stratify_key = (
        pd.Series(treatment).astype(str) + "_" + pd.Series(y).astype(str)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        t_train,
        t_test,
        idx_train,
        idx_test,
    ) = train_test_split(
        X,
        y,
        treatment,
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_key,
    )

    preprocessor = build_preprocessor(numeric, categorical)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "X_train": X_train_proc,
        "X_test": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "t_train": t_train,
        "t_test": t_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "df_full": df,
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
