"""
Feature engineering and transformation pipeline builder.
Separates predictors from targets, audits and drops customer identifiers,
and defines scikit-learn ColumnTransformer pipelines to prevent data leakage.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention.config import config


def get_feature_and_target_columns() -> Tuple[List[str], List[str], str]:
    """Returns numeric predictor columns, categorical predictor columns, and target column."""
    num_cols = list(config.numeric_columns)
    cat_cols = list(config.categorical_columns)
    target_col = config.target_column
    return num_cols, cat_cols, target_col


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer.
    - Numeric features: Median imputation + (optional) StandardScaler.
    - Categorical features: Most-frequent imputation + OneHotEncoder (handle_unknown='ignore').
    """
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drops any undeclared columns like customerID
    )

    return preprocessor


def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts predictor matrix X and target vector y from clean DataFrame.
    Guarantees that customerID, churn_binary, and target columns are excluded from X.
    """
    num_cols, cat_cols, target_col = get_feature_and_target_columns()
    all_features = num_cols + cat_cols

    # Leakage check: confirm ID column is not in feature list
    if config.id_column in all_features:
        raise ValueError(f"Leakage Error: ID column '{config.id_column}' must not be a predictor.")

    X = df[all_features].copy()
    y = (df[target_col] == config.positive_label).astype(int)

    return X, y
