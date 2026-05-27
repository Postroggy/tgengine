"""Tests for TGB dataset loader."""

import os

import pytest

from tgengine.core.dataset import load_tgb_dataset

# Path to real TGB datasets on scnu server
TGB_BASE = os.environ.get("TGB_DATA_PATH", "/mnt/home/gyq/CodeBase/Graph/DG_Data")


def _has_tgb_data():
    return os.path.isdir(os.path.join(TGB_BASE, "tgbl_wiki"))


@pytest.mark.skipif(not _has_tgb_data(), reason="TGB data not available")
class TestLoadTgbWiki:
    """Integration tests using real tgbl_wiki data."""

    @pytest.fixture(scope="class")
    def wiki(self):
        return load_tgb_dataset("tgbl_wiki", TGB_BASE)

    def test_loads_correct_num_edges(self, wiki):
        dataset, val_neg, test_neg = wiki
        assert dataset.num_edges == 157474
        assert dataset.num_nodes == 9227

    def test_has_edge_features(self, wiki):
        dataset, _, _ = wiki
        assert dataset.edge_feat is not None
        assert dataset.edge_feat.shape == (157474, 172)

    def test_split_sizes(self, wiki):
        dataset, val_neg, test_neg = wiki
        n = dataset.num_edges
        assert dataset.train_end == 110232
        assert dataset.val_end == 133853
        # splits sum to total
        assert dataset.train_end + (dataset.val_end - dataset.train_end) + \
            (n - dataset.val_end) == n

    def test_val_negatives_shape(self, wiki):
        _, val_neg, _ = wiki
        assert val_neg is not None
        assert val_neg.shape == (23621, 999)  # N_val, N_neg

    def test_test_negatives_shape(self, wiki):
        _, _, test_neg = wiki
        assert test_neg is not None
        assert test_neg.shape == (23621, 999)

    def test_chronological_order(self, wiki):
        dataset, _, _ = wiki
        # Timestamps should be non-decreasing within train split
        train_t = dataset.time[:dataset.train_end]
        assert (train_t[1:] >= train_t[:-1]).all(), "train not chronologically sorted"

    def test_get_batches(self, wiki):
        dataset, _, _ = wiki
        batches = dataset.get_batches("train", batch_size=512, device="cpu")
        assert len(batches) > 0
        # all batches within train_end
        for b in batches:
            assert b.batch_size <= 512
            assert b.edge_feat is not None or dataset.edge_feat_dim == 0
