"""
Standardized publication-grade chart generation module.
Creates high-resolution figures (300 DPI) for reports and dashboards.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retention.config import config

# Clean, professional styling constants
PALETTE = {
    "primary": "#1E3A8A",      # Deep Navy
    "secondary": "#0284C7",    # Ocean Blue
    "accent": "#EF4444",       # Crimson Red (Churn)
    "retained": "#10B981",     # Emerald Green (Retained)
    "neutral": "#64748B",      # Slate Gray
    "light": "#F8FAFC",        # Off White
}

plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_churn_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plots overall customer churn outcome distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    counts = df[config.target_column].value_counts()
    total = len(df)
    labels = ["Retained (No)", "Churned (Yes)"]
    vals = [counts.get("No", 0), counts.get("Yes", 0)]
    colors = [PALETTE["retained"], PALETTE["accent"]]

    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="none", zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, vals):
        pct = (val / total) * 100
        ax.annotate(
            f"{val:,} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Customer Population Churn Distribution (Snapshot)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Customer Count", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    sns.despine(top=True, right=True)

    out = output_path or (config.figures_dir / "churn_distribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_churn_by_contract(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plots churn prevalence by Contract term with sample sizes."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    grouped = df.groupby("Contract", observed=True)[config.target_column].value_counts(normalize=True).unstack().fillna(0)
    counts = df["Contract"].value_counts()

    contracts = grouped.index.tolist()
    churn_rates = (grouped["Yes"] * 100).tolist()
    colors = [PALETTE["accent"] if r > 30 else PALETTE["secondary"] for r in churn_rates]

    bars = ax.bar(contracts, churn_rates, color=colors, width=0.5, zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, r, contract in zip(bars, churn_rates, contracts):
        n = counts.get(contract, 0)
        ax.annotate(
            f"{r:.1f}%\n(N={n:,})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Churn Prevalence by Contract Type", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Observed Churn Rate (%)", fontsize=10)
    ax.set_ylim(0, max(churn_rates) * 1.25)
    sns.despine(top=True, right=True)

    out = output_path or (config.figures_dir / "churn_by_contract.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_churn_by_tenure_band(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plots churn prevalence across the 5 lifecycle tenure bands."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    grouped = df.groupby("tenure_band", observed=True)[config.target_column].value_counts(normalize=True).unstack().fillna(0)
    counts = df["tenure_band"].value_counts()

    # Reorder according to config
    ordered_bands = [b for b in config.tenure_labels if b in grouped.index]
    rates = [grouped.loc[b, "Yes"] * 100 for b in ordered_bands]

    bars = ax.bar(ordered_bands, rates, color=PALETTE["primary"], width=0.55, zorder=3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, r, band in zip(bars, rates, ordered_bands):
        n = counts.get(band, 0)
        ax.annotate(
            f"{r:.1f}%\n(N={n:,})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Customer Churn Concentration Across Lifecycle Tenure Bands", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Observed Churn Rate (%)", fontsize=10)
    ax.set_ylim(0, max(rates) * 1.25)
    plt.xticks(rotation=15, ha="right")
    sns.despine(top=True, right=True)

    out = output_path or (config.figures_dir / "churn_by_tenure_band.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def plot_tenure_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Plots histogram / KDE of tenure for churned vs retained customers."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    churned = df[df[config.target_column] == "Yes"]["tenure"]
    retained = df[df[config.target_column] == "No"]["tenure"]

    sns.kdeplot(churned, ax=ax, label="Churned Customers", color=PALETTE["accent"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(retained, ax=ax, label="Retained Customers", color=PALETTE["retained"], fill=True, alpha=0.35, linewidth=2)

    ax.set_title("Tenure Density: Churned vs Retained Accounts", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months with Company)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    sns.despine(top=True, right=True)

    out = output_path or (config.figures_dir / "tenure_distribution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def generate_all_descriptive_charts(df: pd.DataFrame) -> List[Path]:
    """Generates and saves all baseline descriptive publication figures."""
    figures = [
        plot_churn_distribution(df),
        plot_churn_by_contract(df),
        plot_churn_by_tenure_band(df),
        plot_tenure_distribution(df),
    ]
    return figures
