"""
Functions for converting, saving and loading SparseGraph to and from different formats.
"""
from typing import Union
import warnings
import pickle
import gzip
import numpy as np

from gust import SparseGraph, GraphCollection


__all__ = [
    'verify_npz',
    'load_from_npz',
    'save_to_npz',
    'load_from_pickle',
    'save_to_pickle'
]


def verify_npz(file_name: str) -> bool:
    """Verify that the file contains one or multiple valid graphs in sparse matrix format.

    Parameters
    ----------
    file_name
        Name of the file containing the sparse graph to verify.

    Returns
    -------
    bool
        True if the npz file contains a valid sparse graph, False otherwise.

    """
    # TODO: implement
    raise NotImplementedError


def load_from_npz(file_name: str) -> Union[SparseGraph, GraphCollection]:
    """Load a SparseGraph or GraphCollection from a Numpy binary file.

    Parameters
    ----------
    file_name
        Name of the file to load.

    Returns
    -------
    gust.SparseGraph or gust.GraphCollection
        Graph(s) in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        if 'type' in loader:
            dataset_type = loader['type']
            del loader['type']
            if dataset_type == SparseGraph.__name__:
                dataset = SparseGraph.from_flat_dict(loader)
            elif dataset_type == GraphCollection.__name__:
                dataset = GraphCollection.from_flat_dict(loader)
            else:
                raise ValueError("Type '{}' of loaded npz-file not recognized.".format(dataset_type))
        else:
            warnings.warn(
                    "Type of saved dataset not specified, using heuristic instead. "
                    "Please update (re-save) your stored graphs.",
                    DeprecationWarning, stacklevel=2)
            if 'dists' in loader.keys():
                dataset = GraphCollection.from_flat_dict(loader)
            else:
                dataset = SparseGraph.from_flat_dict(loader)
    return dataset


def save_to_npz(file_name: str, dataset: Union[SparseGraph, GraphCollection]):
    """Save a SparseGraph or GraphCollection to a Numpy binary file.

    Better (faster) than pickle for single graphs, where disk space is no issue.
    npz doesn't support compression and fails for files larger than 4GiB.

    Parameters
    ----------
    file_name
        Name of the output file.
    dataset
        Graph(s) in sparse matrix format.

    """
    data_dict = dataset.to_flat_dict()
    data_dict['type'] = dataset.__class__.__name__
    np.savez(file_name, **data_dict)


def load_from_pickle(file_name: str, compression: bool = False) -> Union[SparseGraph, GraphCollection]:
    """Load a SparseGraph or GraphCollection from a pickled flattened dictionary.

    Parameters
    ----------
    file_name
        Name of the file to load.
    compression
        Whether to use gzip for compression.

    Returns
    -------
    gust.SparseGraph or gust.GraphCollection
        Graph(s) in sparse matrix format.

    """
    if compression:
        with gzip.open(file_name, 'rb') as f:
            data_dict = pickle.load(f)
    else:
        with open(file_name, 'rb') as f:
            data_dict = pickle.load(f)

    dataset_type = data_dict['type']
    del data_dict['type']

    if dataset_type == SparseGraph.__name__:
        dataset = SparseGraph.from_flat_dict(data_dict)
    elif dataset_type == GraphCollection.__name__:
        dataset = GraphCollection.from_flat_dict(data_dict)
    else:
        raise ValueError("Type '{}' of loaded pickle-file not recognized.".format(dataset_type))
    return dataset


def save_to_pickle(file_name: str, dataset: Union[SparseGraph, GraphCollection], compression: bool = False):
    """Pickle the flattened dictionary of a SparseGraph or GraphCollection.

    Better (smaller and faster) than npz for GraphCollections with many graphs.
    Supports files larger than 4GiB (as opposed to npz).

    Parameters
    ----------
    file_name
        Name of the output file.
    dataset
        Graph(s) in sparse matrix format.
    compression
        Whether to use gzip for compression.

    """
    data_dict = dataset.to_flat_dict()
    data_dict['type'] = dataset.__class__.__name__

    if compression:
        with gzip.open(file_name, 'wb') as f:
            pickle.dump(data_dict, f, protocol=4)
    else:
        with open(file_name, 'wb') as f:
            pickle.dump(data_dict, f, protocol=4)
