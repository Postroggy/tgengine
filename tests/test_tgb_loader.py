"""Tests for TGB dataset loader."""

import os

import pytest

from tgengine.core.dataset import load_dataset

# Path to real TGB datasets on scnu server
TGB_BASE = os.environ.get("TGB_DATA_PATH", "/mnt/home/gyq/CodeBase/Graph/DG_Data")


def _has_tgb_data():
    return (os.path.isdir(os.path.join(TGB_BASE, "tgbl_wiki"))
            or os.path.isdir(os.path.join(TGB_BASE, "tgbl-wiki")))


@pytest.mark.skipif(not _has_tgb_data(), reason="TGB data not available")
class TestLoadTgbWiki:
    """Integration tests using real tgbl-wiki data."""

    @pytest.fixture(scope="class")
    def wiki(self):
        return load_dataset("tgbl-wiki", TGB_BASE)

    def test_loads_correct_num_edges(self, wiki):
        assert wiki.num_edges == 157474
        assert wiki.num_nodes == 8227

    def test_has_edge_features(self, wiki):
        assert wiki.edge_feat is not None
        assert wiki.edge_feat.shape == (157474, 172)

    def test_split_sizes(self, wiki):
        n = wiki.num_edges
        assert wiki.train_end == 110232
        assert wiki.val_end == 133853
        assert wiki.train_end + (wiki.val_end - wiki.train_end) + \
            (n - wiki.val_end) == n

    def test_val_negatives_shape(self, wiki):
        assert wiki.val_neg_candidates is not None
        assert wiki.val_neg_candidates.shape == (23621, 999)

    def test_test_negatives_shape(self, wiki):
        assert wiki.test_neg_candidates is not None
        assert wiki.test_neg_candidates.shape == (23621, 999)

    def test_chronological_order(self, wiki):
        train_t = wiki.time[:wiki.train_end]
        assert (train_t[1:] >= train_t[:-1]).all(), "train not chronologically sorted"

    def test_get_batches(self, wiki):
        batches = wiki.get_batches("train", batch_size=512, device="cpu")
        assert len(batches) > 0
        for b in batches:
            assert b.batch_size <= 512
            assert b.edge_feat is not None or wiki.edge_feat_dim == 0
