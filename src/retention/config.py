"""
Configuration module for Customer Retention & Churn Intelligence.
Loads and validates settings from configs/config.yaml with resilient path resolution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def find_project_root() -> Path:
    """Finds project root directory by searching upward for configs/config.yaml or pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class AppConfig:
    """Encapsulates configuration parameters and paths for the retention package."""

    def __init__(self, config_dict: Dict[str, Any], root_path: Path):
        self.raw = config_dict
        self.root = root_path

        # Project
        self.project_name: str = config_dict.get("project", {}).get("name", "customer-retention-churn")
        self.random_seed: int = int(config_dict.get("project", {}).get("random_seed", 42))

        # Data paths
        data_cfg = config_dict.get("data", {})
        self.raw_path: Path = self.root / data_cfg.get("raw_path", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        self.processed_parquet: Path = self.root / data_cfg.get("processed_parquet", "data/processed/customers_clean.parquet")
        self.processed_csv: Path = self.root / data_cfg.get("processed_csv", "data/processed/customers_clean.csv")
        self.id_column: str = data_cfg.get("id_column", "customerID")
        self.target_column: str = data_cfg.get("target_column", "Churn")
        self.positive_label: str = data_cfg.get("positive_label", "Yes")
        self.negative_label: str = data_cfg.get("negative_label", "No")

        # Columns
        col_cfg = config_dict.get("columns", {})
        self.numeric_columns: List[str] = col_cfg.get("numeric", ["tenure", "MonthlyCharges", "TotalCharges"])
        self.categorical_columns: List[str] = col_cfg.get(
            "categorical",
            [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod"
            ]
        )

        # Analysis
        analysis_cfg = config_dict.get("analysis", {})
        self.tenure_column: str = analysis_cfg.get("tenure_column", "tenure")
        self.monthly_charge_column: str = analysis_cfg.get("monthly_charge_column", "MonthlyCharges")
        self.tenure_bins: List[int] = analysis_cfg.get("tenure_bins", [0, 6, 12, 24, 48, 1000])
        self.tenure_labels: List[str] = analysis_cfg.get(
            "tenure_labels",
            ["0-6m (Onboarding)", "7-12m (Early)", "13-24m (Adolescent)", "25-48m (Mature)", "49+m (Loyal)"]
        )

        # Modeling
        model_cfg = config_dict.get("model", {})
        self.test_size: float = float(model_cfg.get("test_size", 0.20))
        self.cv_folds: int = int(model_cfg.get("cv_folds", 5))
        self.threshold: float = float(model_cfg.get("threshold", 0.50))
        self.candidates: List[str] = model_cfg.get("candidates", ["dummy", "logistic_regression", "hist_gradient_boosting"])

        # Outputs
        out_cfg = config_dict.get("outputs", {})
        self.reports_dir: Path = self.root / out_cfg.get("reports_dir", "reports")
        self.figures_dir: Path = self.root / out_cfg.get("figures_dir", "reports/figures")
        self.tables_dir: Path = self.root / out_cfg.get("tables_dir", "reports/tables")
        self.models_dir: Path = self.root / out_cfg.get("models_dir", "models")


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Loads and validates project configuration from YAML."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(data, PROJECT_ROOT)


# Pre-instantiated default configuration instance
config = load_config()
