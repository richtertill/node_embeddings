"""
Standard preprocessing of SparseGraph objects before further usage.
"""
from numbers import Number
from typing import Tuple, Union
import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.decomposition import PCA

import gust

__all__ = [
    'binarize_labels',
    'construct_laplacian',
    'create_subgraph',
    'reorder_nodes',
    'remove_low_degree_nodes',
    'remove_self_loops',
    'pca_on_attributes',
    'largest_connected_components',
    'sparsegraph_to_networkx',
    'networkx_to_sparsegraph'
]


def binarize_labels(
        labels: np.ndarray,
        sparse_output: bool = False,
        return_classes: bool = False
        ) -> Tuple[Union[np.ndarray, sp.csr_matrix], np.ndarray]:
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels
        Array of node labels in categorical single- or multi-label format. Shape [num_samples]
    sparse_output
        Whether return the label_matrix in CSR format.
    return_classes
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    np.ndarray or sp.csr_matrix
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
        Shape [num_samples, num_classes]
    np.ndarray
        Classes that correspond to each column of the label_matrix. Shape [num_classes]

    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def construct_laplacian(A: sp.spmatrix, type: str = 'unnormalized') -> sp.csr_matrix:
    """Construct Laplacian of a graph given by an adjacency matrix.

    Parameters
    ----------
    A
        Symmetric adjacency matrix in scipy sparse format.
    type
        One of {'unnormalized', 'random_walk', 'symmetrized'}, default 'unnormalized'.
        Type of the Laplacian to compute.
        unnormalized = D - A
        random_walk = I - D^{-1} A
        symmetrized = I - D^{-1/2} A D^{-1/2}

    Returns
    -------
    sp.csr_matrix
        Laplacian matrix in the same format as A.

    """
    if (A != A.T).sum() != 0:
        warnings.warn("Adjacency matrix is not symmetric, the Laplacian might not be PSD.")
    # Make sure that there are no self-loops
    A.setdiag(0)
    A.eliminate_zeros()

    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    if type == 'unnormalized':
        L = sp.diags(D) - A
    elif type == 'random_walk':
        L = sp.eye(num_nodes, dtype=A.dtype) - A / D[:, None]
    elif type == 'symmetrized':
        D_sqrt = np.sqrt(D)
        L = sp.eye(num_nodes, dtype=A.dtype) - A / D_sqrt[:, None] / D_sqrt[None, :]
    else:
        raise ValueError("Unsupported Laplacian type {}.".format(type))
    return L


def create_subgraph(
        sparse_graph: 'gust.SparseGraph',
        _sentinel: None = None,
        nodes_to_remove: np.ndarray = None,
        nodes_to_keep: np.ndarray = None
        ) -> 'gust.SparseGraph':
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    The subgraph partially points to the old graph's data.

    Parameters
    ----------
    sparse_graph
        Input graph.
    _sentinel
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove
        Indices of nodes that have to removed.
    nodes_to_keep
        Indices of nodes that have to be kept.

    Returns
    -------
    gust.SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...).")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is None:
        attr_matrix = None
    else:
        attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.edge_attr_matrix is None:
        edge_attr_matrix = None
    else:
        old_idx = sparse_graph.get_edgeid_to_idx_array()
        keep_edge_idx = np.where(np.all(np.isin(old_idx, nodes_to_keep), axis=1))[0]
        edge_attr_matrix = sparse_graph.edge_attr_matrix[keep_edge_idx]
    if sparse_graph.labels is None:
        labels = None
    else:
        labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is None:
        node_names = None
    else:
        node_names = sparse_graph.node_names[nodes_to_keep]
    # TODO: add warnings / logging
    # print("Resulting subgraph with N = {0}, E = {1}"
    #               .format(sparse_graph.num_nodes(), sparse_graph.num_edges()))
    return gust.SparseGraph(
            adj_matrix, attr_matrix, edge_attr_matrix, labels, node_names,
            sparse_graph.attr_names, sparse_graph.edge_attr_names,
            sparse_graph.class_names, sparse_graph.metadata)


def reorder_nodes(sparse_graph: 'gust.SparseGraph', ordering: np.ndarray):
    """Reorder the nodes in a graph.
    """
    # TODO Implement method
    raise NotImplementedError


def remove_low_degree_nodes(sparse_graph: 'gust.SparseGraph', min_degree: int = 1) -> 'gust.SparseGraph':
    """Remove nodes whose degree is below given threshold.

    If graph is directed, degree = in_degree + out_degree.
    If graph is undirected, min_degree is doubled to account for bidirectional edges.

    Parameters
    ----------
    sparse_graph
        Input graph.
    min_degree
        Minimum degree of nodes that stay in the graph.
        min_degree = 1 - remove isolated nodes.
        min_degree = 2 - remove dangling nodes.

    Returns
    -------
    gust.SparseGraph
        Group with low-degree nodes removed.

    """
    if not sparse_graph.is_directed():
        min_degree = 2 * min_degree  # in undirected graphs each edge is counted twice

    nodes_to_keep = np.where((sparse_graph.adj_matrix.sum(0).A1 + sparse_graph.adj_matrix.sum(1).A1) >= min_degree)[0]
    print("Removing nodes with degree < {0}".format(min_degree))
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def remove_self_loops(sparse_graph: 'gust.SparseGraph') -> 'gust.SparseGraph':
    """Remove self loops (diagonal entries in the adjacency matrix).

    Changes are returned in a partially new SparseGraph.

    """
    num_self_loops = (~np.isclose(sparse_graph.adj_matrix.diagonal(), 0)).sum()
    if num_self_loops > 0:
        adj_matrix = sparse_graph.adj_matrix.copy().tolil()
        adj_matrix.setdiag(0)
        adj_matrix = adj_matrix.tocsr()
        if sparse_graph.edge_attr_matrix is None:
            edge_attr_matrix = None
        else:
            old_idx = sparse_graph.get_edgeid_to_idx_array()
            keep_edge_idx = np.where((old_idx[:, 0] - old_idx[:, 1]) != 0)[0]
            edge_attr_matrix = sparse_graph._edge_attr_matrix[keep_edge_idx]
        warnings.warn("{0} self loops removed".format(num_self_loops))
        return gust.SparseGraph(
                adj_matrix, sparse_graph.attr_matrix, edge_attr_matrix,
                sparse_graph.labels, sparse_graph.node_names,
                sparse_graph.attr_names, sparse_graph.edge_attr_names,
                sparse_graph.class_names, sparse_graph.metadata)
    else:
        return sparse_graph


def pca_on_attributes(sparse_graph: 'gust.SparseGraph', n_components: Union[int, float]) -> 'gust.SparseGraph':
    """Perform PCA on attributes.

    If the attribute matrix is sparse, it is converted to dense and a warning is raised.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        If int, number of components to keep.
        If float, fraction of variance to preserve.

    Returns
    -------
    gust.SparseGraph
        Graph with converted attributes.

    """
    if sparse_graph.attr_matrix is None:
        raise ValueError("The given SparseGraph is not attributed.")

    if sp.isspmatrix(sparse_graph.attr_matrix):
        warnings.warn("Attribute matrix is converted to dense when performing PCA")
        attr_matrix = sparse_graph.attr_matrix.todense()
    else:
        attr_matrix = sparse_graph.attr_matrix

    pca = PCA(n_components=n_components)
    attr_matrix = pca.fit_transform(attr_matrix)
    return gust.SparseGraph(
            sparse_graph.adj_matrix, attr_matrix,
            sparse_graph.edge_attr_matrix, sparse_graph.labels,
            sparse_graph.node_names, None, sparse_graph.edge_attr_names,
            sparse_graph.class_names, sparse_graph.metadata)


def largest_connected_components(sparse_graph: 'gust.SparseGraph', n_components: int = 1) -> 'gust.SparseGraph':
    """Select the largest connected components in the graph.

    Changes are returned in a partially new SparseGraph.

    Parameters
    ----------
    sparse_graph
        Input graph.
    n_components
        Number of largest connected components to keep.

    Returns
    -------
    gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    # TODO: add warnings / logging
    # print("Selecting {0} largest connected components".format(n_components))
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def sparsegraph_to_networkx(sp_graph: 'gust.SparseGraph') -> Union[nx.Graph, nx.DiGraph]:
    """Convert gust SparseGraph to NetworkX graph.

    Everything except metadata is preserved.

    Parameters
    ----------
    sp_graph
        Graph to convert.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Converted graph.

    """
    # Basic graph
    if sp_graph.is_directed():
        nx_graph = nx.DiGraph(sp_graph.adj_matrix)
    else:
        nx_graph = nx.Graph(sp_graph.adj_matrix)

    # Node attributes
    if sp.issparse(sp_graph.attr_matrix):
        for inode, node_attrs in enumerate(sp_graph.attr_matrix):
            for iattr in node_attrs.nonzero()[1]:
                if sp_graph.attr_names is None:
                    nx_graph.nodes[inode][iattr] = node_attrs[iattr]
                else:
                    nx_graph.nodes[inode][sp_graph.attr_names[iattr]] = node_attrs[0, iattr]
    elif isinstance(sp_graph.attr_matrix, np.ndarray):
        for inode, node_attrs in enumerate(sp_graph.attr_matrix):
            for iattr, attr in enumerate(node_attrs):
                if sp_graph.attr_names is None:
                    nx_graph.nodes[inode][iattr] = attr
                else:
                    nx_graph.nodes[inode][sp_graph.attr_names[iattr]] = attr

    # Edge attributes
    if sp.issparse(sp_graph.edge_attr_matrix):
        for edge_id, (i, j) in enumerate(sp_graph.get_edgeid_to_idx_array()):
            row = sp_graph.edge_attr_matrix[edge_id, :]
            for iattr in row.nonzero()[1]:
                if sp_graph.edge_attr_names is None:
                    nx_graph.edges[i, j][iattr] = row[0, iattr]
                else:
                    nx_graph.edges[i, j][sp_graph.edge_attr_names[iattr]] = row[iattr]
    elif isinstance(sp_graph.edge_attr_matrix, np.ndarray):
        for edge_id, (i, j) in enumerate(sp_graph.get_edgeid_to_idx_array()):
            for iattr, attr in enumerate(sp_graph.edge_attr_matrix[edge_id, :]):
                if sp_graph.edge_attr_names is None:
                    nx_graph.edges[i, j][iattr] = attr
                else:
                    nx_graph.edges[i, j][sp_graph.edge_attr_names[iattr]] = attr

    # Labels
    if sp_graph.labels is not None:
        for inode, label in enumerate(sp_graph.labels):
            if sp_graph.class_names is None:
                nx_graph.nodes[inode]['label'] = label
            else:
                nx_graph.nodes[inode]['label'] = sp_graph.class_names[label]

    # Node names
    if sp_graph.node_names is not None:
        mapping = dict(enumerate(sp_graph.node_names))
        if 'self' in sp_graph.attr_names:
            nx_graph = nx.relabel_nodes(nx_graph, mapping)
        else:
            nx.relabel_nodes(nx_graph, mapping, copy=False)

    # Metadata
    if sp_graph.metadata is not None:
        warnings.warn("Could not convert Metadata since NetworkX does not support arbitrary Metadata.")

    return nx_graph


def networkx_to_sparsegraph(
        nx_graph: Union[nx.Graph, nx.DiGraph],
        label_name: str = None,
        sparse_node_attrs: bool = True,
        sparse_edge_attrs: bool = True
        ) -> 'gust.SparseGraph':
    """Convert NetworkX graph to gust SparseGraph.

    Node and edge attributes need to be numeric.
    Missing entries are interpreted as 0.
    Labels can be any object. If non-numeric they are interpreted as
    categorical and enumerated.

    Parameters
    ----------
    nx_graph
        Graph to convert.

    Returns
    -------
    gust.SparseGraph
        Converted graph.

    """
    # Extract node names
    int_names = True
    for node in nx_graph.nodes:
        int_names &= isinstance(node, int)
    if int_names:
        node_names = None
    else:
        node_names = np.array(nx_graph.nodes)
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    # Extract adjacency matrix
    adj_matrix = nx.adjacency_matrix(nx_graph)

    # Collect all node attribute names
    attrs = set()
    for _, node_data in nx_graph.nodes().data():
        attrs.update(node_data.keys())

    # Initialize labels and remove them from the attribute names
    if label_name is None:
        labels = None
    else:
        if label_name not in attrs:
            raise ValueError("No attribute with label name '{}' found.".format(label_name))
        attrs.remove(label_name)
        labels = [0 for _ in range(nx_graph.number_of_nodes())]

    if len(attrs) > 0:
        # Save attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in attrs))
        if all_integer:
            attr_names = None
            attr_mapping = None
        else:
            attr_names = np.array(list(attrs))
            attr_mapping = {k: i for i, k in enumerate(attr_names)}

        # Initialize attribute matrix
        if sparse_node_attrs:
            attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
        else:
            attr_matrix = np.zeros((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
    else:
        attr_matrix = None
        attr_names = None

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == label_name:
                labels[inode] = val
            else:
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError("Node {} has attribute '{}' with value '{}', which is not a number."
                                         .format(inode, key, val))
                    else:
                        raise ValueError("Node '{}' has attribute '{}' with value '{}', which is not a number."
                                         .format(node_names[inode], key, val))
                if attr_mapping is None:
                    attr_matrix[inode, key] = val
                else:
                    attr_matrix[inode, attr_mapping[key]] = val
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    # Convert labels to integers
    if labels is None:
        class_names = None
    else:
        try:
            labels = np.array(labels, dtype=np.float32)
            class_names = None
        except ValueError:
            class_names = np.unique(labels)
            class_mapping = {k: i for i, k in enumerate(class_names)}
            labels_int = np.empty(nx_graph.number_of_nodes(), dtype=np.float32)
            for inode, label in enumerate(labels):
                labels_int[inode] = class_mapping[label]
            labels = labels_int

    # Collect all edge attribute names
    edge_attrs = set()
    for _, _, edge_data in nx_graph.edges().data():
        edge_attrs.update(edge_data.keys())
    if 'weight' in edge_attrs:
        edge_attrs.remove('weight')

    if len(edge_attrs) > 0:
        # Save edge attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in edge_attrs))
        if all_integer:
            edge_attr_names = None
            edge_attr_mapping = None
        else:
            edge_attr_names = np.array(list(edge_attrs))
            edge_attr_mapping = {k: i for i, k in enumerate(edge_attr_names)}

        # Initialize edge attribute matrix
        if sparse_edge_attrs:
            edge_attr_matrix = sp.lil_matrix((adj_matrix.nnz, len(edge_attr_names)), dtype=np.float32)
        else:
            edge_attr_matrix = np.zeros((adj_matrix.nnz, len(edge_attr_names)), dtype=np.float32)
    else:
        edge_attr_matrix = None
        edge_attr_names = None

    # Fill edge attribute matrix
    edgeid_mat = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
    for i, j, edge_attrs in nx_graph.edges.data():
        for key, val in edge_attrs.items():
            if key != 'weight':
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError("Edge {}->{} has attribute '{}' with value '{}', which is not a number."
                                         .format(i, j, key, val))
                    else:
                        raise ValueError("Edge '{}'->'{}' has attribute '{}' with value '{}', which is not a number."
                                         .format(node_names[i], node_names[j], key, val))
                new_key = key if attr_mapping is None else edge_attr_mapping[key]
                edge_attr_matrix[edgeid_mat[i, j], new_key] = val
                if not nx_graph.is_directed():
                    edge_attr_matrix[edgeid_mat[j, i], new_key] = val
    if edge_attr_matrix is not None and sparse_edge_attrs:
        edge_attr_matrix = edge_attr_matrix.tocsr()

    return gust.SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, edge_attr_matrix=edge_attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, edge_attr_names=edge_attr_names, class_names=class_names,
            metadata=None)
