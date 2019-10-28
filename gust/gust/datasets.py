import os
import warnings
from typing import Dict, List, Union
from pathlib import Path
import numpy as np

from .io import load_from_npz, load_from_pickle

import gust


__all__ = [
    'load_dataset',
    'list_datasets',
    'get_dataset_info',
]

data_dir = Path(__file__).parent / Path('data')


def load_dataset(name: str,
                 directory: Union[Path, str] = data_dir
                 ) -> Union[gust.SparseGraph, gust.GraphCollection]:
    """Load a dataset from the GUST collection.

    Parameters
    ----------
    name
        Name of the dataset to load. See available datasets using `gust.datasets.list_datasets`.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    gust.SparseGraph or gust.GraphCollection
        The requested dataset in sparse format.

    """
    if isinstance(directory, str):
        directory = Path(directory)
    if os.path.splitext(name)[1] == '':
        all_filenames = os.listdir(directory)
        possible_filenames = [filename for filename in all_filenames
                              if (name == os.path.splitext(filename)[0]
                                  or name + ".pkl" == os.path.splitext(filename)[0])]
        if len(possible_filenames) == 0:
            raise ValueError("{} doesn't exist.\n(See available datasets using `gust.list_datasets`)"
                             .format(directory / name))
        elif len(possible_filenames) == 1:
            name = possible_filenames[0]
        else:
            warnings.warn("Multiple candidates found for the filename '{}' ({}). Using '{}'."
                          .format(name, possible_filenames, possible_filenames[0]))
            name = possible_filenames[0]
    path_to_file = directory / name
    if path_to_file.exists():
        ext = os.path.splitext(name)[1]
        if ext == '.npz':
            return load_from_npz(path_to_file)
        elif ext == '.pkl':
            return load_from_pickle(path_to_file)
        elif ext == '.gz':
            return load_from_pickle(path_to_file, compression=True)
        else:
            raise ValueError("Unrecognized file extension: '{}'.".format(ext))
    else:
        raise ValueError("{} doesn't exist.\n(See available datasets using `gust.list_datasets`)".format(path_to_file))


def list_datasets(directory: Union[Path, str] = data_dir) -> List[str]:
    """List names of the available datasets in the given directory.

    Parameters
    ----------
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    list
        List of basenames of gust datasets in the given directory.

    """
    if isinstance(directory, str):
        directory = Path(directory)

    spgraph_files = [f for f in directory.iterdir() if f.suffix in ['.npz', '.pkl', '.gz']]
    basenames = [f.stem for f in spgraph_files]
    true_basenames = [os.path.splitext(name)[0] if name.endswith(".pkl") else name for name in basenames]
    return sorted(set(true_basenames))


def get_dataset_info(name: str = None,
                     directory: Union[Path, str] = data_dir
                     ) -> Dict[str, Dict[str, Union[int, bool]]]:
    """Collect basic statistics for the dataset(s).

    Parameters
    ----------
    name
        Name of the dataset to load. See available datasets using `gust.datasets.list_datasets`.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    dict
        Dict, where each entry is a dict containing statistics for the respective datset.

    """
    def collect_info(sparse_graph):
        info = dict()
        info['num_nodes'] = sparse_graph.num_nodes()
        info['num_edges'] = sparse_graph.num_edges()

        if sparse_graph.labels is not None:
            info['num_classes'] = len(np.unique(sparse_graph.labels))
        if sparse_graph.attr_matrix is not None:
            info['num_attr'] = sparse_graph.attr_matrix.shape[1]

        info['has_node_names'] = sparse_graph.node_names is not None
        info['has_attr_names'] = sparse_graph.attr_names is not None
        info['has_class_names'] = sparse_graph.class_names is not None
        return info

    if name is None:
        # Collect information for all datasets
        info_list = dict()
        for dataset in list_datasets(directory):
            sparse_graph = load_dataset(dataset, directory)
            info_list[dataset] = collect_info(sparse_graph)
        return info_list
    else:
        # Collect information only for the requested dataset
        sparse_graph = load_dataset(name, directory)
        return {name: collect_info(sparse_graph)}
