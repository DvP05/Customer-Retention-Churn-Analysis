"""
Model evaluation, calibration, and capacity-threshold analysis module.
Computes comprehensive held-out test metrics, threshold sweeps for operational review,
and exports publication-grade diagnostic plots and prioritized review queues.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from retention.config import config
from retention.reporting.charts import PALETTE


def evaluate_held_out_test(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates fitted pipeline on held-out test data at specified decision threshold."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "test_samples": len(y_test),
        "actual_churned": int(y_test.sum()),
        "predicted_churned": int(y_pred.sum()),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def compute_threshold_capacity_tradeoffs(
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Sweeps thresholds from 0.10 to 0.90 to evaluate operational capacity trade-offs."""
    records = []
    total = len(y_test)
    total_positives = int(y_test.sum())

    for thresh in np.arange(0.10, 0.95, 0.05):
        t = round(float(thresh), 2)
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        flagged = int(y_pred.sum())
        flagged_pct = (flagged / total * 100) if total > 0 else 0.0
        prec = (tp / flagged) if flagged > 0 else 0.0
        rec = (tp / total_positives) if total_positives > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        records.append({
            "threshold": t,
            "review_queue_volume": flagged,
            "review_share_pct": round(flagged_pct, 1),
            "caught_churn_tp": int(tp),
            "false_alarms_fp": int(fp),
            "missed_churn_fn": int(fn),
            "precision_pct": round(prec * 100, 2),
            "recall_pct": round(rec * 100, 2),
            "f1_score": round(f1, 4),
        })

    return pd.DataFrame(records)


def plot_confusion_matrix(cm_dict: Dict[str, int], output_path: Optional[Path] = None) -> Path:
    """Plots publication-grade confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    matrix = np.array([
        [cm_dict["true_negatives"], cm_dict["false_positives"]],
        [cm_dict["false_negatives"], cm_dict["true_positives"]],
    ])

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Retained", "Predicted Churned"],
        yticklabels=["Actual Retained", "Actual Churned"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title("Held-Out Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    out = output_path or (config.figures_dir / "confusion_matrix.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_roc_and_pr_curves(
    y_test: pd.Series,
    y_prob: np.ndarray,
    output_roc: Optional[Path] = None,
    output_pr: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generates ROC and Precision-Recall curves."""
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5, label=f"Candidate Model (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.5, linestyle="--", label="Random Chance Baseline")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    sns.despine()

    roc_path = output_roc or (config.figures_dir / "roc_curve.png")
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    baseline_prev = y_test.mean()

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.plot(rec, prec, color=PALETTE["secondary"], lw=2.5, label=f"Candidate Model (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline_prev, color=PALETTE["accent"], linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_prev:.1%})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall (Coverage of Churned Customers)", fontsize=10)
    ax.set_ylabel("Precision (Accuracy of Flagged Churn)", fontsize=10)
    ax.set_title("Precision-Recall Curve (Critical for Class Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    sns.despine()

    pr_path = output_pr or (config.figures_dir / "precision_recall_curve.png")
    pr_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return roc_path, pr_path


def plot_calibration(y_test: pd.Series, y_prob: np.ndarray, output_path: Optional[Path] = None) -> Path:
    """Plots probability calibration reliability curve."""
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.plot(prob_pred, prob_true, marker="o", color=PALETTE["primary"], lw=2, label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["neutral"], label="Perfect Calibration")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean Predicted Risk Probability", fontsize=10)
    ax.set_ylabel("Fraction of True Churners", fontsize=10)
    ax.set_title("Probability Calibration Reliability Curve", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True)
    sns.despine()

    out = output_path or (config.figures_dir / "calibration_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_threshold_capacity(df_tradeoffs: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plots precision, recall, and review volume across thresholds."""
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax2 = ax1.twinx()

    l1 = ax1.plot(df_tradeoffs["threshold"], df_tradeoffs["precision_pct"], color=PALETTE["primary"], lw=2.2, label="Precision (%)")
    l2 = ax1.plot(df_tradeoffs["threshold"], df_tradeoffs["recall_pct"], color=PALETTE["accent"], lw=2.2, label="Recall (%)")
    l3 = ax2.plot(df_tradeoffs["threshold"], df_tradeoffs["review_queue_volume"], color=PALETTE["neutral"], lw=1.8, linestyle=":", label="Queue Volume (N)")

    ax1.set_xlabel("Decision Threshold (tau)", fontsize=10)
    ax1.set_ylabel("Rate (%)", fontsize=10)
    ax2.set_ylabel("Customer Review Queue Size (Accounts)", fontsize=10, color=PALETTE["neutral"])
    ax1.set_title("Operational Capacity vs Precision/Recall Trade-off", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    out = output_path or (config.figures_dir / "threshold_capacity_curve.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main() -> None:
    interim_dir = config.root / "data" / "interim"
    test_file = interim_dir / "test.parquet"
    model_file = config.models_dir / "churn_pipeline.joblib"

    if not test_file.exists() or not model_file.exists():
        raise FileNotFoundError(
            "Test data or trained model missing. Please run `python -m retention.modeling.train` first."
        )

    print(f"Loading held-out test data from {test_file}...")
    test_df = pd.read_parquet(test_file)
    pipeline = joblib.load(model_file)

    # Separate X and y; ensure ID is not in X
    feature_cols = config.numeric_columns + config.categorical_columns
    X_test = test_df[feature_cols].copy()
    y_test = test_df["Churn"].astype(int)

    print(f"Evaluating model on {len(X_test)} test records at threshold={config.threshold}...")
    metrics = evaluate_held_out_test(pipeline, X_test, y_test, threshold=config.threshold)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Save metrics JSON
    metrics_file = config.reports_dir / "model_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Threshold analysis
    print("Computing capacity trade-offs across decision thresholds...")
    tradeoffs_df = compute_threshold_capacity_tradeoffs(y_test, y_prob)
    tradeoffs_file = config.tables_dir / "threshold_capacity_tradeoff.csv"
    tradeoffs_df.to_csv(tradeoffs_file, index=False)

    # Prioritized Customer Risk Queue
    test_df["churn_probability"] = y_prob
    test_df["expected_charge_exposure"] = test_df["churn_probability"] * test_df["MonthlyCharges"]
    queue_cols = [
        config.id_column, "churn_probability", "MonthlyCharges",
        "expected_charge_exposure", "Contract", "InternetService",
        "tenure", "Churn"
    ]
    queue_df = test_df[queue_cols].sort_values("expected_charge_exposure", ascending=False)
    queue_file = config.tables_dir / "customer_risk_queue.csv"
    queue_df.to_csv(queue_file, index=False)

    # Generate diagnostic plots
    print("Generating publication evaluation figures...")
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_and_pr_curves(y_test, y_prob)
    plot_calibration(y_test, y_prob)
    plot_threshold_capacity(tradeoffs_df)

    print("=" * 60)
    print("HELD-OUT TEST EVALUATION SUMMARY")
    print("=" * 60)
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-Score:           {metrics['f1']:.4f}")
    print(f"Brier Score:        {metrics['brier_score']:.4f}")
    print(f"Confusion Matrix:   TP={metrics['confusion_matrix']['true_positives']}, FP={metrics['confusion_matrix']['false_positives']}, FN={metrics['confusion_matrix']['false_negatives']}, TN={metrics['confusion_matrix']['true_negatives']}")
    print("=" * 60)
    print(f"Saved metrics:     {metrics_file}")
    print(f"Saved trade-offs:  {tradeoffs_file}")
    print(f"Saved review queue: {queue_file}")


if __name__ == "__main__":
    main()
