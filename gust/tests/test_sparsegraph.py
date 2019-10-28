import numpy as np
import scipy.sparse as sp
import pytest

from gust import SparseGraph


class TestSparseGraph:
    def setup(self):
        self.A = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 0. ],
                 [0.5, 0. , 1. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))

    def test_wrong_edge_attr_dim(self):
        for i in range(np.product(self.A.shape) + 2):
            if i != self.A.nnz:
                with pytest.raises(ValueError):
                    SparseGraph(self.A, edge_attr_matrix=np.arange(i))
            else:
                SparseGraph(self.A, edge_attr_matrix=np.arange(i))

    def test_to_undirected(self):
        spA = SparseGraph(self.A)
        spA.to_undirected()
        resA = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 1. ],
                 [0.5, 1. , 1. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 2. , 0. ]]))
        assert np.allclose(spA.adj_matrix.A, resA.A)

    def test_to_undirected_contradiction(self):
        self.A[0, 2] = 0.2
        spA = SparseGraph(self.A)
        with pytest.raises(ValueError):
            spA.to_undirected()

    def test_to_undirected_edgeattrs(self):
        edge_attrs = np.array([0, 1, 2, 3, 1, 5, 6, 7, 8])
        spA = SparseGraph(self.A, edge_attr_matrix=edge_attrs)
        spA.to_undirected()
        resA = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 1. ],
                 [0.5, 1. , 1. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 2. , 0. ]]))
        res_edge_attrs = np.array([0, 1, 2, 3, 8, 1, 3, 5, 6, 6, 7, 8, 7])
        assert np.allclose(spA.adj_matrix.A, resA.A)
        assert np.allclose(spA.edge_attr_matrix, res_edge_attrs)

    def test_to_undirected_edgeattrs_contradiction(self):
        edge_attrs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        spA = SparseGraph(self.A, edge_attr_matrix=edge_attrs)
        with pytest.raises(ValueError):
            spA.to_undirected()

    def test_standardize(self):
        spA = SparseGraph(self.A)
        spA.standardize()
        resA = sp.csr_matrix(np.array(
                [[0. , 0. , 1. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [1. , 1. , 0. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [0. , 1. , 0. , 1. , 0. ]]))
        assert np.allclose(spA.adj_matrix.A, resA.A)

    def test_standardize_edgeattrs(self):
        edge_attrs = np.array([0, 1, 2, 3, 1, 5, 6, 7, 8])
        spA = SparseGraph(self.A, edge_attr_matrix=edge_attrs)
        spA.standardize()
        resA = sp.csr_matrix(np.array(
                [[0. , 0. , 1. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [1. , 1. , 0. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [0. , 1. , 0. , 1. , 0. ]]))
        res_edge_attrs = np.array([1, 3, 8, 1, 3, 6, 6, 7, 8, 7])
        assert np.allclose(spA.adj_matrix.A, resA.A)
        assert np.allclose(spA.edge_attr_matrix, res_edge_attrs)

    def test_flat_dict(self):
        node_attrs = sp.csr_matrix(np.array(
                [[0. , 3. , 2. ],
                 [0. , 0. , 4. ],
                 [1. , 1. , 0. ],
                 [0. , 0. , 1. ],
                 [0. , 2. , 0. ]]))
        attr_names = np.array(['a', 'b', 'c'])
        edge_attrs = np.array(
                [[0 , 1. ],
                 [1 , 0. ],
                 [2 , 1. ],
                 [3 , 0. ],
                 [1 , 0. ],
                 [5 , 4. ],
                 [6 , 0. ],
                 [7 , 3. ],
                 [8 , 2. ]])
        edge_attr_names = np.array(['ae', 'be'])
        labels = np.array([0, 1, 1, 0, 2])
        class_names = np.array(['in', 'between', 'out'])
        spA = SparseGraph(
                self.A, attr_matrix=node_attrs, edge_attr_matrix=edge_attrs,
                attr_names=attr_names, edge_attr_names=edge_attr_names,
                labels=labels, class_names=class_names)
        d = spA.to_flat_dict()
        spB = SparseGraph.from_flat_dict(d)
        assert np.allclose(spA.adj_matrix.A, spB.adj_matrix.A)
        assert (spA.attr_matrix != spB.attr_matrix).nnz == 0
        assert np.allclose(spA.edge_attr_matrix, spB.edge_attr_matrix)
        assert np.allclose(spA.labels, spB.labels)
        assert all((a_name == b_name for a_name, b_name
                    in zip(spA.attr_names, spB.attr_names)))
        assert all((a_name == b_name for a_name, b_name
                    in zip(spA.edge_attr_names, spB.edge_attr_names)))
        assert all((a_name == b_name for a_name, b_name
                    in zip(spA.class_names, spB.class_names)))
