import os
from pathlib import Path
import pytest
import numpy as np
import scipy.sparse as sp

import gust

test_path = Path.cwd()


def test_load_dataset_npz():
    spgraph = gust.SparseGraph(sp.csr_matrix(np.arange(16).reshape(4, 4) > 3, dtype=np.float32))
    gust.io.save_to_npz(test_path / "test.npz", spgraph)
    spgraph2 = gust.load_dataset("test", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    spgraph2 = gust.load_dataset("test.npz", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    os.remove(test_path / "test.npz")


def test_load_dataset_pkl():
    spgraph = gust.SparseGraph(sp.csr_matrix(np.arange(16).reshape(4, 4) > 3, dtype=np.float32))
    gust.io.save_to_pickle(test_path / "test.pkl", spgraph)
    spgraph2 = gust.load_dataset("test", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    spgraph2 = gust.load_dataset("test.pkl", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    os.remove(test_path / "test.pkl")


def test_load_dataset_gz():
    spgraph = gust.SparseGraph(sp.csr_matrix(np.arange(16).reshape(4, 4) > 3, dtype=np.float32))
    gust.io.save_to_pickle(test_path / "test.pkl.gz", spgraph, compression=True)
    spgraph2 = gust.load_dataset("test", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    spgraph2 = gust.load_dataset("test.pkl.gz", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    os.remove(test_path / "test.pkl.gz")


def test_load_dataset_multiple():
    spgraph = gust.SparseGraph(sp.csr_matrix(np.arange(16).reshape(4, 4) > 3, dtype=np.float32))
    gust.io.save_to_pickle(test_path / "test.pkl", spgraph)
    gust.io.save_to_pickle(test_path / "test.pkl.gz", spgraph, compression=True)
    with pytest.warns(UserWarning):
        spgraph2 = gust.load_dataset("test", test_path)
    assert (spgraph.adj_matrix - spgraph2.adj_matrix).nnz == 0
    os.remove(test_path / "test.pkl")
    os.remove(test_path / "test.pkl.gz")


def test_list_datasets():
    filenames = ["test1.bla", "test1.pkl", "test1.pkl.gz", "test3.npz", "test2.pkl.gz", "test4"]
    for filename in filenames:
        Path.touch(test_path / filename)
    datasets = gust.list_datasets(test_path)
    assert len(datasets) == 3
    for filename in filenames:
        os.remove(test_path / filename)
