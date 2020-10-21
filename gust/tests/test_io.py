import os
import numpy as np
import scipy.sparse as sp

from gust import SparseGraph, GraphCollection
from gust.io import load_from_npz, save_to_npz


def test_save_load_edgeattrs():
    A = sp.csr_matrix(np.array(
            [[1. , 0. , 0.5, 0. , 0. ],
             [0. , 1. , 1. , 0. , 0. ],
             [0.5, 0. , 1. , 0. , 0. ],
             [0. , 0. , 1. , 0. , 2. ],
             [0. , 1. , 0. , 0. , 0. ]]))
    edge_attrs = np.array([0, 1, 2, 3, 1, 5, 6, 7, 8])
    spA = SparseGraph(A, edge_attr_matrix=edge_attrs)
    test_file = "test_edgeattrs.npz"
    save_to_npz(test_file, spA)
    spB = load_from_npz(test_file)
    os.remove(test_file)
    assert np.allclose(spA.adj_matrix.A, spB.adj_matrix.A)
    assert np.allclose(spA.edge_attr_matrix, spB.edge_attr_matrix)


def test_save_load_graph_collection():
    A = sp.csr_matrix(np.array(
            [[1. , 0. , 0.5, 0. , 0. ],
             [0. , 1. , 1. , 0. , 0. ],
             [0.5, 0. , 1. , 0. , 0. ],
             [0. , 0. , 1. , 0. , 2. ],
             [0. , 1. , 0. , 0. , 0. ]]))
    edge_attrs_A = np.array([0, 1, 2, 3, 1, 5, 6, 7, 8])
    spA = SparseGraph(A, edge_attr_matrix=edge_attrs_A)
    B = sp.csr_matrix(np.array(
            [[0. , 0. , 0.2, 0. , 0. ],
             [0. , 1. , 0. , 1. , 0. ],
             [0.5, 0. , 0. , 1. , 0. ],
             [0. , 0. , 1. , 0. , 0. ],
             [1. , 0. , 0. , 2. , 0. ]]))
    edge_attrs_B = np.array([7, 6, 5, 4, 3, 2, 1, 0])
    spB = SparseGraph(B, edge_attr_matrix=edge_attrs_B)
    dist_mat = sp.csr_matrix(np.array(
            [[0. , 1. ],
             [1. , 0. ]]))
    gcollA = GraphCollection([spA, spB], dists=dist_mat, metadata="test")
    test_file = "test_graph_collection.npz"
    save_to_npz(test_file, gcollA)
    gcollB = load_from_npz(test_file)
    os.remove(test_file)
    assert np.allclose(gcollA.graphs[0].adj_matrix.A,
                       gcollB.graphs[0].adj_matrix.A)
    assert np.allclose(gcollA.graphs[0].edge_attr_matrix,
                       gcollB.graphs[0].edge_attr_matrix)
    assert np.allclose(gcollA.graphs[1].adj_matrix.A,
                       gcollB.graphs[1].adj_matrix.A)
    assert np.allclose(gcollA.graphs[1].edge_attr_matrix,
                       gcollB.graphs[1].edge_attr_matrix)
    assert (gcollA.dists != gcollB.dists).nnz == 0
    assert gcollA.metadata == gcollB.metadata


def test_load_citeseer():
    citeseer = load_from_npz('../gust/data/citeseer.npz')
    assert citeseer.num_nodes() == 3312
    assert citeseer.num_edges() == 4715
    assert citeseer.adj_matrix.shape == (3312, 3312)
    assert citeseer.attr_matrix.shape == (3312, 3703)
    assert citeseer.labels.shape == (3312,)
    assert citeseer.node_names.shape == (3312,)
    assert citeseer.class_names.shape == (6,)
