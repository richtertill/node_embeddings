import copy
import numpy as np
import scipy.sparse as sp
import pytest

from gust import SparseGraph, GraphCollection, DistanceMatrix


class TestGraphCollection:
    def setup(self):
        A = sp.csr_matrix(np.array(
                [[1. , 0. , 0.5, 0. , 0. ],
                 [0. , 1. , 1. , 0. , 0. ],
                 [0.5, 0. , 1. , 0. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 1. , 0. , 0. , 0. ]]))
        edge_attrs_A = np.array([0, 1, 2, 3, 1, 5, 6, 7, 8])
        self.spA = SparseGraph(A, edge_attr_matrix=edge_attrs_A)
        B = sp.csr_matrix(np.array(
                [[0. , 0. , 0.2, 0. , 0. ],
                 [0. , 1. , 0. , 1. , 0. ],
                 [0.5, 0. , 0. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 0. ],
                 [1. , 0. , 0. , 2. , 0. ]]))
        edge_attrs_B = np.array([7, 6, 5, 4, 3, 2, 1, 0])
        self.spB = SparseGraph(B, edge_attr_matrix=edge_attrs_B)
        dist_mat = np.array(
                [[0. , 1. ],
                 [1. , 0. ]])
        self.gcoll = GraphCollection([self.spA, self.spB], dist_mat)
        dist_spmat = sp.csr_matrix(dist_mat)
        self.gcoll_spdist = GraphCollection([self.spA, self.spB], dist_spmat)

        C = sp.csr_matrix(np.array(
                [[0. , 0. , 0.2, 0. ],
                 [0. , 1. , 0. , 1. ],
                 [0. , 3. , 1. , 0. ],
                 [1. , 0. , 0. , 2. ]]))
        edge_attrs_C = np.array([0, 6, 1, 5, 2, 4, 3])
        self.spC = SparseGraph(C, edge_attr_matrix=edge_attrs_C)

    def test_append(self):
        self.gcoll.append(self.spC)
        assert np.allclose(self.gcoll[2].adj_matrix.A, self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 1. , 0. ],
                 [1. , 0. , 0. ],
                 [0. , 0. , 0. ]])
        assert np.all(self.gcoll.dists == DistanceMatrix(new_dist))

    def test_append_sparse(self):
        self.gcoll_spdist.append(self.spC)
        assert np.allclose(self.gcoll_spdist[2].adj_matrix.A,
                           self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 1. , 0. ],
                 [1. , 0. , 0. ],
                 [0. , 0. , 0. ]])
        assert self.gcoll_spdist.dists.issparse
        assert np.all(self.gcoll_spdist.dists == DistanceMatrix(new_dist))

    def test_add(self):
        new_coll = self.gcoll + GraphCollection(self.spC)
        assert np.allclose(new_coll[2].adj_matrix.A, self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 1. , 0. ],
                 [1. , 0. , 0. ],
                 [0. , 0. , 0. ]])
        assert np.all(new_coll.dists == DistanceMatrix(new_dist))
        new_coll2 = GraphCollection(self.spC) + self.gcoll
        assert np.allclose(new_coll2[0].adj_matrix.A, self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 0. , 0. ],
                 [0. , 0. , 1. ],
                 [0. , 1. , 0. ]])
        assert np.all(new_coll2.dists == DistanceMatrix(new_dist))

    def test_add_sparse(self):
        new_coll = self.gcoll_spdist + GraphCollection(self.spC)
        assert np.allclose(new_coll[2].adj_matrix.A, self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 1. , 0. ],
                 [1. , 0. , 0. ],
                 [0. , 0. , 0. ]])
        assert new_coll.dists.issparse
        assert np.all(new_coll.dists == DistanceMatrix(new_dist))

    def test_extend_single(self):
        self.gcoll.extend(GraphCollection(self.spC))
        assert np.allclose(self.gcoll[2].adj_matrix.A, self.spC.adj_matrix.A)
        new_dist = np.array(
                [[0. , 1. , 0. ],
                 [1. , 0. , 0. ],
                 [0. , 0. , 0. ]])
        assert np.all(self.gcoll.dists == DistanceMatrix(new_dist))

    def test_extend(self):
        self.gcoll.extend(copy.deepcopy(self.gcoll))
        new_dist = np.array(
                [[0. , 1. , 0. , 0. ],
                 [1. , 0. , 0. , 0. ],
                 [0. , 0. , 0. , 1. ],
                 [0. , 0. , 1. , 0. ]])
        assert np.all(self.gcoll.dists == DistanceMatrix(new_dist))

    def test_del(self):
        self.gcoll.append(self.spC)
        del self.gcoll[0]
        assert len(self.gcoll) == 2
        assert np.allclose(self.gcoll[0].adj_matrix.A, self.spB.adj_matrix.A)
        assert np.allclose(self.gcoll[1].adj_matrix.A, self.spC.adj_matrix.A)
        assert np.all(self.gcoll.dists == DistanceMatrix(np.zeros([2, 2])))

    def test_del_sparse(self):
        self.gcoll_spdist.append(self.spC)
        del self.gcoll_spdist[0]
        assert len(self.gcoll_spdist) == 2
        assert np.allclose(self.gcoll_spdist[0].adj_matrix.A, self.spB.adj_matrix.A)
        assert np.allclose(self.gcoll_spdist[1].adj_matrix.A, self.spC.adj_matrix.A)
        assert self.gcoll_spdist.dists.issparse
        assert np.all(self.gcoll_spdist.dists == DistanceMatrix(np.zeros([2, 2])))

    def test_faulty_dist_matrices(self):
        # self-distance not zero
        dist_mat = np.array(
                [[1. , 1. ],
                 [1. , 0. ]])
        with pytest.raises(ValueError):
            GraphCollection([self.spA, self.spB], dist_mat)
        # distance not symmetric
        dist_mat = np.array(
                [[0. , 2. ],
                 [1. , 0. ]])
        with pytest.raises(ValueError):
            GraphCollection([self.spA, self.spB], dist_mat)
        # distance negative
        dist_mat = np.array(
                [[ 0. ,-1. ],
                 [-1. , 0. ]])
        with pytest.raises(ValueError):
            GraphCollection([self.spA, self.spB], dist_mat)
        # Matrix too large
        dist_mat = np.array(
                [[ 0. ,-1. , 0. ],
                 [ 0. , 0. , 0. ],
                 [-1. , 0. , 0. ]])
        with pytest.raises(ValueError):
            GraphCollection([self.spA, self.spB], dist_mat)

    def test_set_dist(self):
        self.gcoll.dists[0, 1] = 2
        assert self.gcoll.dists[1, 0] == 2
        with pytest.raises(ValueError):
            self.gcoll.dists[0, 0] = 1
        with pytest.raises(ValueError):
            self.gcoll.dists[0, 1] = -1

    def test_iteration(self):
        for i, graph in enumerate(self.gcoll):
            if i == 0:
                assert (graph.adj_matrix != self.spA.adj_matrix).nnz == 0
            elif i == 1:
                assert (graph.adj_matrix != self.spB.adj_matrix).nnz == 0


class TestDistanceMatrix:
    def setup(self):
        self.A = sp.csr_matrix(np.array(
                [[0. , 0. , 0.5, 0. , 0. ],
                 [0. , 0. , 1. , 0. , 0. ],
                 [0.5, 1. , 0. , 1. , 0. ],
                 [0. , 0. , 1. , 0. , 2. ],
                 [0. , 0. , 0. , 2. , 0. ]]))
        self.dA = DistanceMatrix(self.A)
        self.B = sp.csr_matrix(np.array(
                [[0. , 0. , 0.2, 0. , 1. ],
                 [0. , 0. , 0. , 1. , 0. ],
                 [0.2, 0. , 0. , 1. , 3. ],
                 [0. , 1. , 1. , 0. , 0. ],
                 [1. , 0. , 3. , 0. , 0. ]]))
        self.dB = DistanceMatrix(self.B)

    def test_dist_add(self):
        dC = self.dA + self.dB
        resC = sp.csr_matrix(np.array(
                [[0. , 0. , 0.7, 0. , 1. ],
                 [0. , 0. , 1. , 1. , 0. ],
                 [0.7, 1. , 0. , 2. , 3. ],
                 [0. , 1. , 2. , 0. , 2. ],
                 [1. , 0. , 3. , 2. , 0. ]]))
        assert (dC != DistanceMatrix(resC)).nnz == 0

    def test_dist_add_iadd(self):
        dC = self.dA + self.dB
        self.dA += self.dB
        assert (dC != self.dA).nnz == 0

    def test_dist_sub_zero(self):
        dC = self.dA - self.dA
        assert np.all(dC == DistanceMatrix(np.zeros([5, 5])))

    def test_dist_sub_consistency(self):
        dC = self.dA + self.dB
        dD = dC - self.dB
        dE = dC - self.dA
        assert (dD != self.dA).nnz == 0
        assert np.allclose(dE._matrix.A, self.dB._matrix.A)

    def test_dist_isub_consistency(self):
        dC = self.dA + self.dB
        dD = copy.deepcopy(dC)
        dC -= self.dB
        dD -= self.dA
        assert (dC != self.dA).nnz == 0
        assert np.allclose(dD._matrix.A, self.dB._matrix.A)

    def test_sub_larger(self):
        with pytest.raises(ValueError):
            self.dA - self.dB
        with pytest.raises(ValueError):
            self.dA -= self.dB
