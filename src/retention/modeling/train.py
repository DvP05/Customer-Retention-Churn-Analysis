"""
Machine learning training and model selection pipeline.
Implements stratified splitting, 5-fold cross-validation across baseline and candidate models,
leakage-free ColumnTransformer pipelines, and model serialization.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from retention.config import config
from retention.features.build import build_preprocessor, extract_features_and_target


def create_model_pipeline(model_name: str) -> Pipeline:
    """Builds a scikit-learn Pipeline encapsulating preprocessing and estimator."""
    num_cols = list(config.numeric_columns)
    cat_cols = list(config.categorical_columns)

    if model_name == "dummy":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        estimator = DummyClassifier(strategy="most_frequent")
    elif model_name == "logistic_regression":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.random_seed)
    elif model_name == "hist_gradient_boosting":
        # HistGradientBoosting works directly with scaled or unscaled features
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        estimator = HistGradientBoostingClassifier(random_state=config.random_seed)
    elif model_name == "random_forest":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        estimator = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=config.random_seed)
    else:
        raise ValueError(f"Unknown candidate model: {model_name}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return pipeline


def cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_names: list[str],
) -> pd.DataFrame:
    """Evaluates multiple candidate models via 5-fold Stratified Cross-Validation."""
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_seed)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_acc": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    records = []
    for name in candidate_names:
        pipe = create_model_pipeline(name)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        records.append({
            "model": name,
            "mean_pr_auc": round(float(scores["test_pr_auc"].mean()), 4),
            "std_pr_auc": round(float(scores["test_pr_auc"].std()), 4),
            "mean_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
            "std_roc_auc": round(float(scores["test_roc_auc"].std()), 4),
            "mean_f1": round(float(scores["test_f1"].mean()), 4),
            "mean_balanced_acc": round(float(scores["test_balanced_acc"].mean()), 4),
            "mean_accuracy": round(float(scores["test_accuracy"].mean()), 4),
        })

    cv_df = pd.DataFrame(records).sort_values("mean_pr_auc", ascending=False)
    return cv_df


def train_and_persist_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fits the chosen pipeline on the full training set and saves serialized artifacts."""
    pipeline = create_model_pipeline(best_model_name)
    pipeline.fit(X_train, y_train)

    models_dir = config.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / "churn_pipeline.joblib"
    joblib.dump(pipeline, model_file)

    metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "train_samples": len(X_train),
        "positive_class_ratio": round(float(y_train.mean()), 4),
        "numeric_features": config.numeric_columns,
        "categorical_features": config.categorical_columns,
        "model_file": str(model_file),
    }

    meta_file = models_dir / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return pipeline, metadata


def main() -> None:
    parquet_path = config.processed_parquet
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {parquet_path}.")

    print(f"Loading clean dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print("Extracting feature matrix X and target y (excluding customerID)...")
    X, y = extract_features_and_target(df)

    print(f"Splitting data into 80% train and 20% test (Stratified, seed={config.random_seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_seed,
    )

    # Save interim train/test splits for testing and evaluation
    interim_dir = config.root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train.rename("Churn")], axis=1)
    test_df = pd.concat([X_test, y_test.rename("Churn")], axis=1)

    # Also keep customerID in test set for review queue demonstration
    test_df[config.id_column] = df.loc[test_df.index, config.id_column]

    train_df.to_parquet(interim_dir / "train.parquet", index=False)
    test_df.to_parquet(interim_dir / "test.parquet", index=False)

    print(f"Train set: {len(X_train):,} rows | Held-out test set: {len(X_test):,} rows")

    print("\nRunning 5-fold Stratified Cross-Validation on training folds...")
    candidates = config.candidates
    cv_summary = cross_validate_candidates(X_train, y_train, candidates)

    # Export CV table
    table_dir = config.tables_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    cv_file = table_dir / "model_comparison_metrics.csv"
    cv_summary.to_csv(cv_file, index=False)

    print("=" * 60)
    print("CROSS-VALIDATION BENCHMARK RESULTS")
    print("=" * 60)
    print(cv_summary.to_string(index=False))
    print("=" * 60)

    # Choose best model (e.g., highest PR-AUC)
    best_name = cv_summary.iloc[0]["model"]
    print(f"\nFitting selected candidate '{best_name}' on full training set...")
    pipeline, meta = train_and_persist_best_model(X_train, y_train, best_name)

    print(f"Model successfully saved to {meta['model_file']}")


if __name__ == "__main__":
    main()
