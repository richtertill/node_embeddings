"""
General utilities including numpy extensions and graph utils.
"""
from typing import Iterable, List, Union
import numba
import numpy as np
import scipy.sparse as sp
import warnings

from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.model_selection import train_test_split


__all__ = [
    'cartesian_product',
    'edges_to_sparse',
    'train_val_test_split_adjacency',
    'train_val_test_split_tabular',
    'sort_nodes',
    'sparse_feeder',
    'gumbel_sample_random_walks',
    'edge_cover',
    'construct_line_graph',
    'sample_random_walks_per_node',
    'sample_random_walks_numba',
]


def cartesian_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Form the cartesian product (i.e. all pairs of values) between two arrays.

    Parameters
    ----------
    x
        Left array in the cartesian product. Shape [Nx]
    y
        Right array in the cartesian product. Shape [Ny]

    Returns
    -------
    np.ndarray
        Cartesian product. Shape [Nx * Ny]

    """
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def edges_to_sparse(edges: np.ndarray, num_nodes: int, weights: np.ndarray = None) -> sp.csr_matrix:
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    Parameters
    ----------
    edges
        Array with each row storing indices of an edge as (u, v). Shape [num_edges, 2]
    num_nodes
        Number of nodes in the resulting graph.
    weights
        Weights of the edges. If None, all edges weights are set to 1. Shape [num_edges]

    Returns
    -------
    sp.csr_matrix
        Adjacency matrix in CSR format.

    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


def train_val_test_split_tabular(
        *arrays: Iterable[Union[np.ndarray, sp.spmatrix]],
        train_size: float = 0.5,
        val_size: float = 0.3,
        test_size: float = 0.2,
        stratify: np.ndarray = None,
        random_state: int = None
        ) -> List[Union[np.ndarray, sp.spmatrix]]:
    """Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices with the same length / shape[0].
    train_size
        Proportion of the dataset included in the train split.
    val_size
        Proportion of the dataset included in the validation split.
    test_size
        Proportion of the dataset included in the test split.
    stratify
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state
        Random_state is the seed used by the random number generator;

    Returns
    -------
    list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def train_val_test_split_adjacency(A, p_val=0.10, p_test=0.05, random_state=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """Create edge and non-edge train, validation and test sets.

    Split the edges of the adjacency matrix into train, validation and test edges.
    Randomly sample validation and test non-edges.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    p_val : float
        Percentage of validation edges. Default p_val=0.10
    p_test : float
        Percentage of test edges. Default p_test=0.05
    random_state : int
        Seed for numpy.random. Default seed=0
    neg_mul : int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    every_node : bool
        Make sure each node appears at least once in the train set. Default every_node=True
    connected : bool
        Make sure the training graph is still connected after the split
    undirected : bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    set_ops : bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    asserts : bool
        Unit test like checks. Default asserts=False

    Returns
    -------
    train_ones : array-like, shape [n_train, 2]
        Indices of the train edges
    val_ones : array-like, shape [n_val, 2]
        Indices of the validation edges
    val_zeros : array-like, shape [n_val, 2]
        Indices of the validation non-edges
    test_ones : array-like, shape [n_test, 2]
        Indices of the test edges
    test_zeros : array-like, shape [n_test, 2]
        Indices of the test non-edges

    """
    assert p_val + p_test > 0
    assert A.max() == 1  # no weights
    assert A.min() == 0  # no negative edges
    assert A.diagonal().sum() == 0  # no self-loops
    assert not np.any(A.sum(0).A1 + A.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (A != A.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        A = sp.tril(A).tocsr()  # consider only upper triangular
        A.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(random_state)

    E = A.nnz
    N = A.shape[0]
    s_train = int(E * (1 - p_val - p_test))

    idx = np.arange(N)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert connected_components(A)[0] == 1  # make sure original graph is connected
            A_hold = minimum_spanning_tree(A)
        else:
            A.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = A.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(A)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed is {:.2f}.'
                                     .format(min_size / E))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if N > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed is {:.2f}.'
                                     .format(N / E))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, A[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack((np.row_stack(map(np.random.choice, A[:, d == 0].T.tolil().rows)),
                                                     idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(A[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == N

            A_hold = edges_to_sparse(hold_edges, N)

        A_hold[A_hold > 1] = 1
        A_hold.eliminate_zeros()
        A_sample = A - A_hold

        s_train = s_train - A_hold.nnz
    else:
        A_sample = A

    idx_ones = np.random.permutation(A_sample.nnz)
    ones = np.column_stack(A_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(A_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, N, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, N, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[A[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[np.unique(random_sample[:, 0] * N + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, N, 2)
            if A[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        A = A.maximum(A.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == A.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), N) != A).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or connected_components(A_hold)[0] == 1
        assert not every_node or ((A_hold - A) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def sort_nodes(z, deg=None):
    """Sort the nodes such that consecutive nodes belong to the same cluster.

    Clusters are sorted from smallest to largest.
    Optionally also sort by node degrees within each cluster.

    Parameters
    ----------
    z : array-like, shape [n_samples]
        The cluster indicators (labels)
    deg : array-like, shape [n_samples]
        Degree of each node

    Returns
    -------
    o : array-like, shape [n_samples]
        Indices of the nodes that give the desired sorting

    """
    _, idx, cnts = np.unique(z, return_counts=True, return_inverse=True)
    counts = cnts[idx]

    if deg is None:
        return np.lexsort((z, counts))
    else:
        return np.lexsort((deg, z, counts))


def sparse_feeder(M):
    """Convert a sparse matrix to the format suitable for feeding as a tf.SparseTensor.

    Parameters
    ----------
    M : sp.spmatrix
        Matrix to convert.

    Returns
    -------
    indices : array-like, shape [num_edges, 2]
        Indices of the nonzero elements.
    values : array-like, shape [num_edges]
        Values of the nonzero elements.
    shape : tuple
        Shape of the matrix.

    """
    M = sp.coo_matrix(M)
    return np.vstack((M.row, M.col)).T, M.data, M.shape


def gumbel_sample_random_walks(A, walks_per_node, walk_length, random_state=None):
    """Sample random walks from a given graph using the Gumbel trick.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix
    walks_per_node : int
        The number of random walks from each node.
    walk_length : int
        The length of each random walk.
    random_state : int or None
        Random seed for the numpy RNG.

    Returns
    -------
    random_walks : array-like, shape [N*r, l]
        The sampled random walks

    """
    if random_state is not None:
        np.random.seed(random_state)

    num_nodes = A.shape[0]
    samples = []

    prev_nodes = np.random.permutation(np.repeat(np.arange(num_nodes), walks_per_node))
    samples.append(prev_nodes)

    for _ in range(walk_length - 1):
        A_cur = A[prev_nodes]
        A_cur[A_cur > 0] = np.random.gumbel(loc=3, size=[A_cur.nnz])  # loc=3 so that all samples are bigger than 0

        prev_nodes = A_cur.argmax(1).A1
        samples.append(prev_nodes)

    return np.array(samples).T


def edge_cover(A):
    """
    Approximately compute minimum edge cover.

    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix

    Returns
    -------
    edges : array-like, shape [?, 2]
        The edges the form the edge cover
    """

    N = A.shape[0]
    d_in = A.sum(0).A1
    d_out = A.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((A[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, A[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(A.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == N:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(N), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, A[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((A[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert A[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == N

    return edges


def sample_random_walks_per_node(A, l):
    """
    Sample a single random walk per node from the graph.

    Parameters
    ----------
    A : sp.spmatrix
        Sparse adjacency matrix
    l : int
        Random walk length

    Returns
    -------
    walks : array-like, shape [N, l]
        The sampled random walks

    """
    N = A.shape[0]
    walks = np.zeros([N, l], dtype=np.int)
    walks[:, 0] = np.arange(N)

    for idx in range(1, l):
        walks[:, idx] = np.fromiter(map(np.random.choice, A[walks[:, idx - 1]].rows), dtype=np.int)

    return walks


def construct_line_graph(A):
    """Construct a line graph from an undirected original graph.

    Parameters
    ----------
    A : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.

    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjancy matrix of the line graph.
    """
    N = A.shape[0]
    edges = np.column_stack(sp.triu(A, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]

    I = sp.eye(N).tocsr()
    E1 = I[e1]
    E2 = I[e2]

    L = E1.dot(E1.T) + E1.dot(E2.T) + E2.dot(E1.T) + E2.dot(E2.T)

    return L - 2*sp.eye(L.shape[0])


@numba.jit(nopython=True, parallel=True)
def _rw(indptr, indices, l, r, seed):
    """
    Sample r random walks of length l per node in parallel from the graph.

    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    l : int
        Random walk length
    r : int
        Number of random walks per node
    seed : int
        Random seed

    Returns
    -------
    walks : array-like, shape [r*N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for ir in range(r):
        for n in range(N):
            for il in range(l):
                walks.append(n)
                n = np.random.choice(indices[indptr[n]:indptr[n + 1]])

    return np.array(walks)


@numba.jit(nopython=True, parallel=True)
def _rw_nbr(indptr, indices, l, r, seed):
    """
    Sample r non-backtracking random walks of length l per node in parallel from the graph.

    Parameters
    ----------
    indptr : array-like
        Pointer for the edges of each node
    indices : array-like
        Edges for each node
    l : int
        Random walk length
    r : int
        Number of random walks per node
    seed : int
        Random seed

    Returns
    -------
    walks : array-like, shape [r*N*l]
        The sampled random walks
    """
    np.random.seed(seed)
    N = len(indptr) - 1
    walks = []

    for ir in range(r):
        for n in range(N):
            prev = -1
            for il in range(l):
                walks.append(n)
                nbr = indices[indptr[n]:indptr[n + 1]]

                if len(nbr) > 1 and prev != -1:
                    fbr = list(nbr)
                    fbr.remove(prev)
                    nbr = np.array(fbr)

                prev = n
                n = np.random.choice(nbr)

    return np.array(walks)


def sample_random_walks_numba(A, walk_length, num_walks, non_backtracking=False, seed=0):
    """
    Sample random walks of fixed length from each node in the graph in parallel.

    Parameters
    ----------
    A : sp.spmatrix, shape [num_nodes, num_nodes]
        Sparse adjacency matrix.
    walk_length : int
        Random walk length.
    num_walks : int
        Number of random walks per node.
    non_backtracking : bool
        Whether to make the random walks non-backtracking.
    seed : int
        Random seed.

    Returns
    -------
    walks : np.ndarray, shape [num_walks * num_nodes, walk_length]
        The sampled random walks.

    """
    if non_backtracking:
        return _rw_nbr(A.indptr, A.indices, walk_length, num_walks, seed).reshape([-1, walk_length])
    else:
        return _rw(A.indptr, A.indices, walk_length, num_walks, seed).reshape([-1, walk_length])
