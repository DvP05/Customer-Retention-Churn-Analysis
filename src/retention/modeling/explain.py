"""
Model explainability and permutation feature importance module.
Calculates statistical feature associations on held-out test data,
generates importance figures, and documents non-causal interpretation boundaries.
"""

from pathlib import Path
from typing import Any, Optional
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from retention.config import config
from retention.reporting.charts import PALETTE


def compute_permutation_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Computes permutation importance on held-out test dataset."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=config.random_seed,
        scoring="average_precision",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 12,
    output_path: Optional[Path] = None,
) -> Path:
    """Plots top N features ranked by permutation importance."""
    top_df = importance_df.head(top_n).sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.barh(
        top_df["feature"],
        top_df["importance_mean"],
        xerr=top_df["importance_std"],
        color=PALETTE["primary"],
        edgecolor="none",
        alpha=0.9,
        capsize=3,
        height=0.6,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Decrease in Test PR-AUC (Permutation Importance)", fontsize=10)
    ax.set_title(f"Top {top_n} Predictive Features for Customer Churn", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out = output_path or (config.figures_dir / "feature_importance.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main() -> None:
    interim_dir = config.root / "data" / "interim"
    test_file = interim_dir / "test.parquet"
    model_file = config.models_dir / "churn_pipeline.joblib"

    if not test_file.exists() or not model_file.exists():
        raise FileNotFoundError("Missing test data or pipeline model.")

    print(f"Loading held-out test data from {test_file}...")
    test_df = pd.read_parquet(test_file)
    pipeline = joblib.load(model_file)

    feature_cols = config.numeric_columns + config.categorical_columns
    X_test = test_df[feature_cols].copy()
    y_test = test_df["Churn"].astype(int)

    print("Computing permutation feature importance on held-out test data (scoring=PR-AUC)...")
    imp_df = compute_permutation_importance(pipeline, X_test, y_test)

    # Export table
    table_dir = config.tables_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    out_table = table_dir / "feature_importance.csv"
    imp_df.to_csv(out_table, index=False)

    # Plot
    fig_path = plot_feature_importance(imp_df)

    print("=" * 60)
    print("TOP PREDICTIVE FEATURES (STATISTICAL ASSOCIATIONS)")
    print("=" * 60)
    print(imp_df.head(10).to_string(index=False))
    print("=" * 60)
    print("NON-CAUSAL INTERPRETATION NOTICE:")
    print("These scores represent predictive association strength, not causal mechanisms.")
    print("Observing that 'Contract' has high importance does not prove that forcing customers")
    print("into longer commitments will causally eliminate their desire to cancel.")
    print("=" * 60)
    print(f"Saved table:  {out_table}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
