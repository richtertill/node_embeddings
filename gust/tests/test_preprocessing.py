import pytest
import numpy as np
import scipy.sparse as sp

import gust


class TestPreprocessing:
    def setup(self):
        self.A = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 1. ],
                 [0.5, 0. , 1. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))

    def test_create_subgraph(self):
        spA = gust.SparseGraph(self.A.copy())
        keep = [0, 2, 3]
        spB = gust.create_subgraph(spA, nodes_to_keep=keep)
        # Check that changes are not done in-place
        assert np.allclose(self.A.A, spA.adj_matrix.A)
        B = sp.csr_matrix(np.array(
                [[1. , 0.5, 0. ],
                 [0.5, 1. , 0. ],
                 [0. , 1. , 0. ]]))
        assert np.allclose(B.A, spB.adj_matrix.A)

    def test_create_subgraph_edgeattrs(self):
        edge_attrs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        spA = gust.SparseGraph(self.A, edge_attr_matrix=edge_attrs)
        keep = [0, 2, 3]
        spB = gust.create_subgraph(spA, nodes_to_keep=keep)
        # Check that changes are not done in-place
        assert np.allclose(spA.edge_attr_matrix, edge_attrs)
        B = sp.csr_matrix(np.array(
                [[1. , 0.5, 0. ],
                 [0.5, 1. , 0. ],
                 [0. , 1. , 0. ]]))
        edge_attrs_B = np.array([0, 1, 5, 6, 7])
        assert np.allclose(B.A, spB.adj_matrix.A)
        assert np.allclose(spB.edge_attr_matrix, edge_attrs_B)

    def test_remove_self_loops(self):
        spA = gust.SparseGraph(self.A.copy())
        spB = gust.remove_self_loops(spA)
        # Check that changes are not done in-place
        assert np.allclose(self.A.A, spA.adj_matrix.A)
        B = sp.csr_matrix(np.array(
                [[0. , 0. , 0.5, 0. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [0.5, 0. , 0. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))
        assert np.allclose(B.A, spB.adj_matrix.A)

    def test_remove_self_loops_edgeattrs(self):
        edge_attrs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        spA = gust.SparseGraph(self.A, edge_attr_matrix=edge_attrs)
        spB = gust.remove_self_loops(spA)
        # Check that changes are not done in-place
        assert np.allclose(spA.edge_attr_matrix, edge_attrs)
        B = sp.csr_matrix(np.array(
                [[0. , 0. , 0.5, 0. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [0.5, 0. , 0. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))
        edge_attrs_B = np.array([1, 3, 4, 5, 7, 8, 9])
        assert np.allclose(B.A, spB.adj_matrix.A)
        assert np.allclose(spB.edge_attr_matrix, edge_attrs_B)

    def test_remove_full_self_loops(self):
        self.A += sp.eye(5)
        spA = gust.SparseGraph(self.A.copy())
        spB = gust.remove_self_loops(spA)
        # Check that changes are not done in-place
        assert np.allclose(self.A.A, spA.adj_matrix.A)
        B = sp.csr_matrix(np.array(
                [[0. , 0. , 0.5, 0. , 0. ],
                 [0. , 0. , 1. , 0. , 1. ],
                 [0.5, 0. , 0. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))
        assert np.allclose(B.A, spB.adj_matrix.A)

    def test_sparsegraph_to_from_networkx_simple(self):
        spA = gust.SparseGraph(self.A)
        nx_graph = gust.sparsegraph_to_networkx(spA)
        spB = gust.networkx_to_sparsegraph(nx_graph)
        assert np.allclose(spA.adj_matrix.A, spB.adj_matrix.A)

    @pytest.mark.parametrize('sparse', [True, False])
    def test_sparsegraph_to_from_networkx(self, sparse):

        # Set up original graph
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
                 [8 , 2. ],
                 [9 , 0.3]])
        edge_attr_names = np.array(['ae', 'be'])
        labels = np.array([0, 1, 1, 0, 2])
        class_names = np.array(['in', 'between', 'out'])
        A_sym = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 1. ],
                 [0.5, 1. , 0. , 0. , 0. ],
                 [0. , 0. , 0. , 0. , 2. ],
                 [0. , 1. , 0. , 2. , 0. ]]))
        edge_attrs_sym = np.array(
                [[0 , 1. ],
                 [1 , 0. ],
                 [2 , 1. ],
                 [3 , 0. ],
                 [8 , 2. ],
                 [1 , 0. ],
                 [3 , 0. ],
                 [7 , 3. ],
                 [8 , 2. ],
                 [7 , 3.]])
        for i in range(2):
            A = self.A if i == 0 else A_sym
            edge_a = edge_attrs if i == 0 else edge_attrs_sym
            spA = gust.SparseGraph(
                    A, attr_matrix=node_attrs, edge_attr_matrix=edge_a,
                    attr_names=attr_names, edge_attr_names=edge_attr_names,
                    labels=labels, class_names=class_names)

            # Convert to NetworkX and back
            nx_graph = gust.sparsegraph_to_networkx(spA)
            spB = gust.networkx_to_sparsegraph(
                    nx_graph, label_name='label',
                    sparse_node_attrs=sparse, sparse_edge_attrs=sparse)

            # Check adjacency matrix
            assert np.allclose(spA.adj_matrix.A, spB.adj_matrix.A)

            # Check node attributes
            assert len(spA.attr_names) == len(spB.attr_names)
            for iold, attr in enumerate(spA.attr_names):
                assert len(np.where(spB.attr_names == attr)[0]) == 1
                inew = np.where(spB.attr_names == attr)[0][0]
                if sparse:
                    assert (spA.attr_matrix[:, iold]
                            != spB.attr_matrix[:, inew]).nnz == 0
                else:
                    assert np.allclose(spA.attr_matrix.A[:, iold],
                                    spB.attr_matrix[:, inew])

            # Check edge attributes
            assert len(spA.edge_attr_names) == len(spB.edge_attr_names)
            for iold, attr in enumerate(spA.edge_attr_names):
                assert len(np.where(spB.edge_attr_names == attr)[0]) == 1
                inew = np.where(spB.edge_attr_names == attr)[0][0]
                if sparse:
                    assert np.allclose(spA.edge_attr_matrix[:, iold],
                                    spB.edge_attr_matrix.A[:, inew])
                else:
                    assert np.allclose(spA.edge_attr_matrix[:, iold],
                                    spB.edge_attr_matrix[:, inew])

            # Check labels and class names
            assert len(spA.class_names) == len(spB.class_names)
            class_mapping = {}
            for iold, label in enumerate(spA.class_names):
                assert len(np.where(spB.class_names == label)[0]) == 1
                class_mapping[iold] = np.where(spB.class_names == label)[0][0]
            assert len(spA.labels) == len(spB.labels)
            all((class_mapping[old_label] == spB.labels[i]
                for i, old_label in enumerate(spA.labels)))


def test_largest_connected_components():
    A = sp.csr_matrix(np.array(
            [[1. , 0. , 0.5, 0. , 0. ],
             [0. , 1. , 1. , 0. , 0. ],
             [0.5, 0. , 1. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 2. ],
             [0. , 0. , 0. , 0. , 0. ]]))
    spA = gust.SparseGraph(A.copy())
    spB = gust.largest_connected_components(spA)
    # Check that changes are not done in-place
    assert np.allclose(A.A, spA.adj_matrix.A)
    B = sp.csr_matrix(np.array(
            [[1. , 0. , 0.5 ],
             [0. , 1. , 1.  ],
             [0.5, 0. , 1.  ]]))
    assert np.allclose(B.A, spB.adj_matrix.A)


def test_largest_connected_components_edgeattrs():
    A = sp.csr_matrix(np.array(
            [[1. , 0. , 0.5, 0. , 0. ],
             [0. , 1. , 0. , 0.5, 0. ],
             [0. , 0. , 1. , 0. , 0. ],
             [0. , 0. , 0. , 0. , 2. ],
             [0. , 0. , 0. , 0. , 0. ]]))
    edge_attrs = np.array([0, 1, 2, 3, 4, 5])
    spA = gust.SparseGraph(A, edge_attr_matrix=edge_attrs)
    spB = gust.largest_connected_components(spA)
    # Check that changes are not done in-place
    assert np.allclose(spA.edge_attr_matrix, edge_attrs)
    B = sp.csr_matrix(np.array(
            [[1. , 0.5, 0. ],
             [0. , 0. , 2. ],
             [0. , 0. , 0. ]]))
    edge_attrs_B = np.array([2, 3, 5])
    assert np.allclose(B.A, spB.adj_matrix.A)
    assert np.allclose(spB.edge_attr_matrix, edge_attrs_B)
