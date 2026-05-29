"""Dataset download utilities.

Supports automatic download of standard CTDG benchmark datasets
from publicly available sources:
  - DyGLib/DGB datasets (Zenodo + SNAP/JODIE)
  - TGB link prediction datasets (Compute Canada)
  - TGB-Seq datasets (tgb-seq package hosting)
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================================
# DyGLib / DGB datasets
# ============================================================================

# Original JODIE datasets from SNAP (already in ml_* format via DyGLib GitHub)
_JODIE_CSV_URLS = {
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
    "lastfm": "http://snap.stanford.edu/jodie/lastfm.csv",
}

_DYGLIB_GITHUB = "https://raw.githubusercontent.com/yule-BUAA/DyGLib/main/processed_data"
_DYGLIB_FEATURE_URLS = {
    "wikipedia": f"{_DYGLIB_GITHUB}/wikipedia/ml_wikipedia.npy",
    "reddit": f"{_DYGLIB_GITHUB}/reddit/ml_reddit.npy",
    "mooc": f"{_DYGLIB_GITHUB}/mooc/ml_mooc.npy",
    "lastfm": f"{_DYGLIB_GITHUB}/lastfm/ml_lastfm.npy",
    "uci": f"{_DYGLIB_GITHUB}/uci/ml_uci.npy",
}
_DYGLIB_NODE_FEATURE_URLS = {
    "wikipedia": f"{_DYGLIB_GITHUB}/wikipedia/ml_wikipedia_node.npy",
    "reddit": f"{_DYGLIB_GITHUB}/reddit/ml_reddit_node.npy",
    "mooc": f"{_DYGLIB_GITHUB}/mooc/ml_mooc_node.npy",
    "lastfm": f"{_DYGLIB_GITHUB}/lastfm/ml_lastfm_node.npy",
    "uci": f"{_DYGLIB_GITHUB}/uci/ml_uci_node.npy",
}
_DYGLIB_CSV_URLS = {
    "uci": f"{_DYGLIB_GITHUB}/uci/ml_uci.csv",
}

# DGB datasets from Zenodo (https://zenodo.org/records/7213796)
_ZENODO_BASE = "https://zenodo.org/records/7213796/files"
_ZENODO_DATASETS = {
    # Already-preprocessed (zip contains ml_* files directly)
    "enron": {"url": f"{_ZENODO_BASE}/enron.zip?download=1", "bipartite": None},
    "SocialEvo": {"url": f"{_ZENODO_BASE}/SocialEvo.zip?download=1", "bipartite": None},
    # Need preprocessing from raw CSV (non-bipartite)
    "Flights": {"url": f"{_ZENODO_BASE}/Flights.zip?download=1", "bipartite": False},
    "CanParl": {"url": f"{_ZENODO_BASE}/CanParl.zip?download=1", "bipartite": False},
    "USLegis": {"url": f"{_ZENODO_BASE}/USLegis.zip?download=1", "bipartite": False},
    "UNtrade": {"url": f"{_ZENODO_BASE}/UNtrade.zip?download=1", "bipartite": False},
    "UNvote": {"url": f"{_ZENODO_BASE}/UNvote.zip?download=1", "bipartite": False},
    "Contacts": {"url": f"{_ZENODO_BASE}/Contacts.zip?download=1", "bipartite": False},
}

# ============================================================================
# TGB datasets (link prediction)
# ============================================================================

_TGB_BASE = "https://object-arbutus.cloud.computecanada.ca/tgb"
_TGB_DATASETS = {
    "tgbl-wiki": {"url": f"{_TGB_BASE}/tgbl-wiki-v2.zip", "version": "v2"},
    "tgbl-review": {"url": f"{_TGB_BASE}/tgbl-review-v2.zip", "version": "v2"},
    "tgbl-coin": {"url": f"{_TGB_BASE}/tgbl-coin-v2.zip", "version": "v2"},
    "tgbl-comment": {"url": f"{_TGB_BASE}/tgbl-comment.zip", "version": "v1"},
    "tgbl-flight": {"url": f"{_TGB_BASE}/tgbl-flight-v2.zip", "version": "v2"},
}

# ============================================================================
# TGB-Seq datasets (Hugging Face: https://huggingface.co/TGB-Seq)
# ============================================================================

_TGBSEQ_HF_BASE = os.environ.get(
    "HF_ENDPOINT", "https://huggingface.co"
).rstrip("/") + "/datasets/TGB-Seq"
_TGBSEQ_DATASETS = {
    "tgbseq-ml20m": {"hf_name": "ML-20M"},
    "tgbseq-taobao": {"hf_name": "Taobao"},
    "tgbseq-yelp": {"hf_name": "Yelp"},
    "tgbseq-googlelocal": {"hf_name": "GoogleLocal"},
    "tgbseq-flickr": {"hf_name": "Flickr"},
    "tgbseq-youtube": {"hf_name": "YouTube"},
    "tgbseq-patent": {"hf_name": "Patent"},
    "tgbseq-wikilink": {"hf_name": "WikiLink"},
}

# ============================================================================
# All supported datasets (union)
# ============================================================================

ALL_DATASETS = (
    set(_JODIE_CSV_URLS.keys())
    | set(_DYGLIB_CSV_URLS.keys())
    | set(_ZENODO_DATASETS.keys())
    | set(_TGB_DATASETS.keys())
    | set(_TGBSEQ_DATASETS.keys())
)


# ============================================================================
# Proxy support
# ============================================================================

_proxy: Optional[str] = os.environ.get("TGENGINE_PROXY")


def set_proxy(proxy: Optional[str]) -> None:
    """Set HTTP/HTTPS proxy for all dataset downloads.

    Args:
        proxy: proxy URL (e.g. "http://127.0.0.1:7890", "socks5h://127.0.0.1:1080"),
               or None to disable.

    Can also be set via the TGENGINE_PROXY environment variable.
    Supports HTTP, HTTPS, and SOCKS5 proxies.

    Example::

        from tgengine.utils.download import set_proxy
        set_proxy("http://127.0.0.1:7890")   # clash/v2ray default port
        download_dataset("wikipedia")
    """
    global _proxy
    _proxy = proxy


def _get_opener() -> Optional[urllib.request.OpenerDirector]:
    """Build a URL opener with proxy if configured."""
    proxy = _proxy
    if proxy is None:
        return None
    handler = urllib.request.ProxyHandler({
        "http": proxy,
        "https": proxy,
    })
    return urllib.request.build_opener(handler)


# ============================================================================
# Internal helpers
# ============================================================================


def _download_file(url: str, dest: Path, desc: str = "", retries: int = 3) -> bool:
    """Download a file with progress reporting and retries. Returns True if successful."""
    import subprocess

    for attempt in range(retries):
        try:
            if attempt > 0:
                print(f"  Retry {attempt}/{retries-1}...")
            print(f"  Downloading {desc or url}...")

            # Try wget for non-HF URLs (HF uses redirects that work better with urllib)
            use_wget = (shutil.which("wget") and _proxy is None
                        and "huggingface" not in url and "hf-mirror" not in url)
            if use_wget:
                result = subprocess.run(
                    ["wget", "-q", "--timeout=300", "--tries=3",
                     "-O", str(dest), url],
                    capture_output=True, timeout=3600,
                )
                if result.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                    return True
                if dest.exists():
                    dest.unlink()

            # Fallback to urllib (supports proxy)
            opener = _get_opener()
            if opener is not None:
                with opener.open(url) as resp, open(str(dest), "wb") as f:
                    shutil.copyfileobj(resp, f)
            else:
                urllib.request.urlretrieve(url, str(dest))
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                subprocess.TimeoutExpired) as e:
            if dest.exists():
                dest.unlink()
            if attempt == retries - 1:
                print(f"  Warning: failed to download {url}: {e}")
                return False


def _download_and_extract_zip(url: str, dest_dir: Path, desc: str = "") -> bool:
    """Download a zip and extract into dest_dir. Returns True if successful."""
    zip_path = dest_dir / "_temp.zip"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not _download_file(url, zip_path, desc):
        return False
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(dest_dir))
        return True
    except zipfile.BadZipFile as e:
        print(f"  Error: bad zip file: {e}")
        return False
    finally:
        if zip_path.exists():
            zip_path.unlink()


def _preprocess_dgb_raw(raw_csv: Path, out_dir: Path, bipartite: bool) -> None:
    """Preprocess a DGB raw CSV into ml_*.csv + ml_*.npy + ml_*_node.npy.

    DGB raw CSV format: u, i, ts, label, feat1, feat2, ...
    (comma-separated, first row is header)
    """
    users, items, timestamps, labels = [], [], [], []
    edge_feats_list = []

    with open(raw_csv, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            u, i = int(parts[0]), int(parts[1])
            ts = float(parts[2])
            label = float(parts[3])
            feats = [float(x) for x in parts[4:]] if len(parts) > 4 else []
            users.append(u)
            items.append(i)
            timestamps.append(ts)
            labels.append(label)
            edge_feats_list.append(feats)

    users = np.array(users, dtype=np.int64)
    items = np.array(items, dtype=np.int64)

    # Reindex: bipartite shifts dst IDs to be disjoint from src IDs
    if bipartite:
        items = items + users.max() + 1

    # IDs start from 1 (0 reserved for padding)
    users += 1
    items += 1

    n_edges = len(users)
    feat_dim = len(edge_feats_list[0]) if edge_feats_list and edge_feats_list[0] else 0

    # Edge features: prepend zero row at index 0
    if feat_dim > 0:
        feats = np.array(edge_feats_list, dtype=np.float32)
        feats = np.vstack([np.zeros((1, feat_dim), dtype=np.float32), feats])
    else:
        feats = np.zeros((n_edges + 1, 1), dtype=np.float32)

    # Node features: all zeros (172 dims)
    num_nodes = max(users.max(), items.max()) + 1
    node_feats = np.zeros((num_nodes, 172), dtype=np.float32)

    # Write ml_*.csv
    name = raw_csv.stem  # e.g. "Flights"
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame({
        "u": users,
        "i": items,
        "ts": timestamps,
        "label": labels,
        "idx": np.arange(1, n_edges + 1),
    })
    df.to_csv(out_dir / f"ml_{name}.csv")
    np.save(str(out_dir / f"ml_{name}.npy"), feats)
    np.save(str(out_dir / f"ml_{name}_node.npy"), node_feats)


def _preprocess_tgb_zip(dataset_dir: Path, name: str, version: str) -> None:
    """Ensure TGB zip contents are accessible in the dataset directory.

    TGB zips extract into subdirectories. This function moves all relevant
    files (edgelist CSV, negative sample pkl) to the dataset root directory.
    NO reindexing or format conversion — the loader handles that to preserve
    alignment with pre-computed negative samples.
    """
    # Move all relevant files from subdirectories to dataset root
    for pattern in ["*edgelist*.csv", "*_ns*.pkl", "*node_feat*", "*_edge*"]:
        for f in dataset_dir.rglob(pattern):
            target = dataset_dir / f.name
            if f != target and not target.exists():
                shutil.move(str(f), str(target))

    # Clean up empty subdirectories
    for d in list(dataset_dir.iterdir()):
        if d.is_dir():
            try:
                d.rmdir()  # only removes if empty
            except OSError:
                pass

    # Verify edgelist exists
    name_hyphen = name.replace("_", "-")
    found = list(dataset_dir.glob("*edgelist*.csv"))
    if found:
        print(f"  TGB dataset '{name}' ready: {[f.name for f in found]}")
    else:
        print(f"  Warning: no edgelist CSV found for {name}")


def _preprocess_tgbseq(dataset_dir: Path, name: str, hf_name: str) -> None:
    """Verify TGB-Seq files are present. No conversion needed — loader reads
    the original CSV directly to preserve the split column and node IDs."""
    csv_path = dataset_dir / f"{hf_name}.csv"
    if csv_path.exists():
        print(f"  TGB-Seq dataset '{name}' ready: {csv_path.name}")
    else:
        print(f"  Warning: CSV not found at {csv_path}")


# ============================================================================
# Public API
# ============================================================================


def download_dataset(name: str, dest_dir: str = "datasets") -> Path:
    """Download a dataset if not already present.

    Supports three dataset families:
      - DyGLib/DGB: wikipedia, reddit, mooc, lastfm, uci, enron, SocialEvo,
        Flights, CanParl, USLegis, UNtrade, UNvote, Contacts
      - TGB: tgbl-wiki, tgbl-review, tgbl-coin, tgbl-comment, tgbl-flight
      - TGB-Seq: tgbseq-ml20m, tgbseq-taobao, tgbseq-yelp, tgbseq-googlelocal,
        tgbseq-flickr, tgbseq-youtube, tgbseq-patent, tgbseq-wikilink

    Args:
        name: dataset name (case-sensitive for DGB datasets).
        dest_dir: root directory for datasets.

    Returns:
        Path to the dataset directory.

    Raises:
        ValueError: if dataset name is not recognized.
        RuntimeError: if download fails.
    """
    if name not in ALL_DATASETS:
        available = ", ".join(sorted(ALL_DATASETS))
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}. "
            f"For other datasets, manually place files in {dest_dir}/{name}/."
        )

    base = Path(dest_dir) / name

    # Check if already downloaded (look for ml_*.csv)
    if _is_downloaded(base, name):
        return base

    print(f"Dataset '{name}' not found locally. Downloading...")
    base.mkdir(parents=True, exist_ok=True)

    # Route to appropriate downloader
    if name in _JODIE_CSV_URLS or name in _DYGLIB_CSV_URLS:
        _download_dyglib_direct(name, base)
    elif name in _ZENODO_DATASETS:
        _download_zenodo(name, base)
    elif name in _TGB_DATASETS:
        _download_tgb(name, base)
    elif name in _TGBSEQ_DATASETS:
        _download_tgbseq(name, base)

    print(f"Dataset '{name}' ready at {base}")
    return base


def list_available_datasets() -> list[str]:
    """Return list of all datasets available for auto-download."""
    return sorted(ALL_DATASETS)


def list_datasets_by_family() -> dict[str, list[str]]:
    """Return datasets grouped by family (dyglib, tgb, tgbseq)."""
    return {
        "dyglib": sorted(
            set(_JODIE_CSV_URLS.keys()) | set(_DYGLIB_CSV_URLS.keys()) | set(_ZENODO_DATASETS.keys())
        ),
        "tgb": sorted(_TGB_DATASETS.keys()),
        "tgbseq": sorted(_TGBSEQ_DATASETS.keys()),
    }


# ============================================================================
# Download routing
# ============================================================================


def _is_downloaded(base: Path, name: str) -> bool:
    """Check if dataset already exists locally."""
    if not base.exists():
        return False
    # DyGLib datasets
    if (base / f"ml_{name}.csv").exists():
        return True
    # TGB datasets: edgelist CSV
    if list(base.glob("*edgelist*.csv")):
        return True
    # TGB-Seq datasets: HF CSV
    if name in _TGBSEQ_DATASETS:
        hf_name = _TGBSEQ_DATASETS[name]["hf_name"]
        if (base / f"{hf_name}.csv").exists():
            return True
    return False


def _download_dyglib_direct(name: str, base: Path) -> None:
    """Download JODIE/DyGLib datasets (already in ml_* format on remote)."""
    # Download CSV
    csv_url = _JODIE_CSV_URLS.get(name) or _DYGLIB_CSV_URLS.get(name)
    csv_path = base / f"ml_{name}.csv"
    if not _download_file(csv_url, csv_path, f"ml_{name}.csv"):
        raise RuntimeError(f"Failed to download {name} dataset CSV")

    # Download edge features
    if name in _DYGLIB_FEATURE_URLS:
        npy_path = base / f"ml_{name}.npy"
        _download_file(_DYGLIB_FEATURE_URLS[name], npy_path, f"ml_{name}.npy")

    # Download node features
    if name in _DYGLIB_NODE_FEATURE_URLS:
        node_path = base / f"ml_{name}_node.npy"
        _download_file(_DYGLIB_NODE_FEATURE_URLS[name], node_path, f"ml_{name}_node.npy")


def _download_zenodo(name: str, base: Path) -> None:
    """Download DGB dataset from Zenodo and preprocess if needed."""
    info = _ZENODO_DATASETS[name]
    url = info["url"]
    bipartite = info["bipartite"]

    if not _download_and_extract_zip(url, base, f"{name}.zip"):
        raise RuntimeError(f"Failed to download {name} from Zenodo")

    # Find extracted content — zip may extract into a subfolder
    extracted_dir = base / name
    if not extracted_dir.exists():
        # Maybe extracted directly into base
        extracted_dir = base

    if bipartite is None:
        # Already preprocessed — find and move ml_* files to base
        _move_ml_files(extracted_dir, base, name)
    else:
        # Need preprocessing from raw CSV
        raw_csv = extracted_dir / f"{name}.csv"
        if not raw_csv.exists():
            # Try looking deeper
            candidates = list(base.rglob(f"{name}.csv"))
            if candidates:
                raw_csv = candidates[0]
            else:
                raise RuntimeError(
                    f"Raw CSV not found for {name}. Expected: {raw_csv}"
                )
        _preprocess_dgb_raw(raw_csv, base, bipartite)

    # Clean up extracted subfolder if different from base
    if extracted_dir != base and extracted_dir.exists():
        # Move any remaining useful files (e.g. README)
        for f in extracted_dir.iterdir():
            if f.name.startswith("ml_") and not (base / f.name).exists():
                shutil.move(str(f), str(base / f.name))
        shutil.rmtree(str(extracted_dir), ignore_errors=True)


def _move_ml_files(src_dir: Path, dest_dir: Path, name: str) -> None:
    """Move ml_* files from extracted directory to destination."""
    for pattern in [f"ml_{name}*"]:
        for f in src_dir.rglob(pattern):
            target = dest_dir / f.name
            if f != target:
                shutil.move(str(f), str(target))


def _download_tgb(name: str, base: Path) -> None:
    """Download TGB dataset and convert to our format."""
    info = _TGB_DATASETS[name]
    url = info["url"]
    version = info["version"]

    if not _download_and_extract_zip(url, base, f"{name}.zip"):
        raise RuntimeError(f"Failed to download {name} from TGB")

    _preprocess_tgb_zip(base, name, version)


def _download_tgbseq(name: str, base: Path) -> None:
    """Download TGB-Seq dataset from Hugging Face and convert to our format."""
    info = _TGBSEQ_DATASETS[name]
    hf_name = info["hf_name"]

    # Hugging Face direct file download URL pattern
    hf_resolve = f"{_TGBSEQ_HF_BASE}/{hf_name}/resolve/main"

    # Download CSV
    csv_url = f"{hf_resolve}/{hf_name}.csv"
    csv_path = base / f"{hf_name}.csv"
    if not _download_file(csv_url, csv_path, f"{hf_name}.csv"):
        raise RuntimeError(f"Failed to download {name} CSV from Hugging Face")

    # Download test negative samples (optional)
    ns_url = f"{hf_resolve}/{hf_name}_test_ns.npy"
    ns_path = base / f"{hf_name}_test_ns.npy"
    _download_file(ns_url, ns_path, f"{hf_name}_test_ns.npy")

    _preprocess_tgbseq(base, name, hf_name)
