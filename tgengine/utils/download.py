"""Dataset download utilities.

Supports automatic download of standard CTDG benchmark datasets
from publicly available sources.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASET_URLS = {
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
    "lastfm": "http://snap.stanford.edu/jodie/lastfm.csv",
    "uci": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/uci/ml_uci.csv",
}

FEATURE_URLS = {
    "wikipedia": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/wikipedia/ml_wikipedia.npy",
    "reddit": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/reddit/ml_reddit.npy",
    "mooc": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/mooc/ml_mooc.npy",
    "lastfm": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/lastfm/ml_lastfm.npy",
    "uci": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/uci/ml_uci.npy",
}

NODE_FEATURE_URLS = {
    "wikipedia": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/wikipedia/ml_wikipedia_node.npy",
    "reddit": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/reddit/ml_reddit_node.npy",
    "mooc": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/mooc/ml_mooc_node.npy",
    "lastfm": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/lastfm/ml_lastfm_node.npy",
    "uci": "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data/uci/ml_uci_node.npy",
}


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress reporting. Returns True if successful."""
    try:
        print(f"  Downloading {desc or url}...")
        urllib.request.urlretrieve(url, str(dest))
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  Warning: failed to download {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_dataset(name: str, dest_dir: str = "datasets") -> Path:
    """Download a dataset if not already present.

    Downloads the csv and npy files into {dest_dir}/{name}/ following
    the DyGLib naming convention (ml_{name}.csv, ml_{name}.npy, etc.)

    Args:
        name: dataset name (wikipedia, reddit, mooc, lastfm, uci).
        dest_dir: root directory for datasets.

    Returns:
        Path to the dataset directory.

    Raises:
        ValueError: if dataset name is not recognized.
    """
    if name not in DATASET_URLS:
        available = ", ".join(sorted(DATASET_URLS.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. Available for auto-download: {available}. "
            f"For other datasets, manually place files in {dest_dir}/{name}/."
        )

    base = Path(dest_dir) / name
    csv_path = base / f"ml_{name}.csv"

    if csv_path.exists():
        return base

    print(f"Dataset '{name}' not found locally. Downloading...")
    base.mkdir(parents=True, exist_ok=True)

    # Download CSV
    csv_url = DATASET_URLS[name]
    if not _download_file(csv_url, csv_path, f"ml_{name}.csv"):
        raise RuntimeError(f"Failed to download {name} dataset CSV from {csv_url}")

    # Download edge features
    if name in FEATURE_URLS:
        npy_path = base / f"ml_{name}.npy"
        _download_file(FEATURE_URLS[name], npy_path, f"ml_{name}.npy")

    # Download node features (optional, may not exist for all datasets)
    if name in NODE_FEATURE_URLS:
        node_path = base / f"ml_{name}_node.npy"
        _download_file(NODE_FEATURE_URLS[name], node_path, f"ml_{name}_node.npy")

    print(f"Dataset '{name}' downloaded to {base}")
    return base


def list_available_datasets() -> list[str]:
    """Return list of datasets available for auto-download."""
    return sorted(DATASET_URLS.keys())
