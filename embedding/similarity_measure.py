'''
File to produce following similarity measures:
- Adjacency Matrix A
- Laplacian P=D^−1A
- Deep Walk / Symmetric, Normalized Laplacian D^(−1/2)AD^(−1/2)
- NetMF
- Personalized Page Rank
- Sum of Power of Transitions
Input: Adjacency Matrix of a Graph (Dataset), NxN Matrix
Initialize: None
Output: Similarity Measure, NxN Matrix
'''

import sys
sys.path.insert(0, '../')
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from time import time
from utils import graph_util

# def adjacency(A):
#     # Adjacency Matrix A
#     return torch.FloatTensor(A.toarray()).cuda()

# def laplacian(A):
#     #Laplacian P=D^−1A
#     degreenp=A.sum(axis=1)
#     degree = torch.from_numpy(degreenp).cuda().view(-1)
#     degreematrix = torch.zeros(N,N)
#     degreematrix[np.diag_indices(N)]=degree
#     invdegree = torch.inverse(degreematrix)
#     adj = torch.FloatTensor(A.toarray()).cuda()
#     return torch.matmul(invdegree, adj)

# def dw(A):
#     #Deep Walk / Symmetric, Normalized Laplacian D^(−1/2)AD^(−1/2)
#     degreenp=A.sum(axis=1)
#     degree = torch.from_numpy(degreenp).cuda().view(-1)
#     degreematrix = torch.zeros(N,N)
#     degreematrix[np.diag_indices(N)]=degree
#     sqrtdegree = degreematrix.sqrt()
#     invdegree = torch.inverse(sqrtdegree)
#     adj = torch.FloatTensor(A.toarray()).cuda() 
#     return torch.matmul(invdegree, torch.matmul(adj, invdegree))


def adjacency(A):
    # Adjacency Matrix A
    return graph_util.csr_matrix_to_torch_tensor(A)

def laplacian(A):
    return

def Transition(A):
    #Laplacian P=D^−1A
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    L = sp.diags(D) - A
    return graph_util.csr_matrix_to_torch_tensor(L)

def sym_normalized_laplacian(A):
    #Symmetric, Normalized Laplacian P=D^(−1/2)AD^(−1/2)
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    L = A / D_sqrt[:, None] / D_sqrt[None, :]
    return graph_util.csr_matrix_to_torch_tensor(L)

def NetMF(A):
    return

def PPR(A):
    return

def Sum_Power_Tran(A):
    return

def Sim_Rank(A):
    return