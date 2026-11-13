"""
Data loading and provenance tracking module.
Reads raw subscription dataset and captures cryptographic fingerprint and file metadata.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from retention.config import config


def compute_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file for immutability verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_raw_data(
    input_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV data and extracts file metadata without modifying the source.

    Returns:
        Tuple of (raw DataFrame, metadata dictionary)
    """
    path = Path(input_path) if input_path else config.raw_path
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {path}. Please place the dataset in data/raw/."
        )

    sha256_hash = compute_sha256(path)
    file_stat = path.stat()

    df = pd.read_csv(path)

    metadata = {
        "filename": path.name,
        "filepath": str(path.resolve()),
        "file_size_bytes": file_stat.st_size,
        "sha256": sha256_hash,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }

    return df, metadata


if __name__ == "__main__":
    df, meta = load_raw_data()
    print("=" * 60)
    print("RAW DATASET INGESTION SUMMARY")
    print("=" * 60)
    print(f"File:       {meta['filename']}")
    print(f"SHA-256:    {meta['sha256']}")
    print(f"Size:       {meta['file_size_bytes']:,} bytes")
    print(f"Rows:       {meta['row_count']:,}")
    print(f"Columns:    {meta['column_count']}")
    print("=" * 60)
