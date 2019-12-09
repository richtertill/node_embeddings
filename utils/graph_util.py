# try: import cPickle as pickle
# except: import pickle
# import numpy as np
# import networkx as nx
# import random
# import itertools
# import time
# import pdb

import gust  # library for loading graph data
import torch
import scipy as sp
import numpy as np


def load_dataset(name='cora'):
    A, X, _, y = gust.load_dataset(name).standardize().unpack()
    # A - adjacency matrix 
    # X - attribute matrix - not needed
    # y - node labels

    if (A != A.T).sum() > 0:
        raise RuntimeError("The graph must be undirected!")

    if (A.data != 1).sum() > 0:
        raise RuntimeError("The graph must be unweighted!")
    
    return A,y

def csr_matrix_to_torch_tensor(matrix):
    coo = sp.sparse.coo_matrix(matrix)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


