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

#we mustn't use the D[D == 0] = 1, as it destroys the row stochasticy

def adjacency(A):
    # Adjacency Matrix A
    return graph_util.csr_matrix_to_torch_tensor(A)

def laplacian(A):
    #Transition Matrix P=D-A
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    L = sp.diags(D) - A
    return L

def Transition(A):
    #Laplacian P=D^−1A
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    a=np.ones(D.shape[0])
    D_inv = np.divide(a, D, out=np.zeros_like(a), where=D!=0)
    L = sp.diags(D_inv) * A
    return L

def sym_normalized_laplacian(A):
    #Symmetric, Normalized Laplacian P=D^(−1/2)AD^(−1/2)
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    a=np.ones(D_sqrt.shape[0])
    D_sqrt_inv = np.divide(a, D_sqrt, out=np.zeros_like(a), where=D!=0) 
    L = sp.diags(D_sqrt_inv) * A * sp.diags(D_sqrt_inv)
    #L = A / D_sqrt[:, None] / D_sqrt[None, :]
    return L

def NetMF(A):
    return

def PPR(A):
    #Personalized PageRank Matrix as described in https://openreview.net/pdf?id=H1gL-2A9Ym with the there used hyperparameter alpha=0.1
    #P=alpha(I-(1-alpha)*D^-1/2(A+I)D^-1/2)^-1
    alpha = 0.1  
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    a=np.ones(D_sqrt.shape[0])
    D_sqrt_inv = np.divide(a, D_sqrt, out=np.zeros_like(a), where=D!=0)
    A_tilde = sp.diags(D_sqrt_inv) * (A + sp.identity(A.shape[0])) * sp.diags(D_sqrt_inv)
    L_inv = (sp.identity(A.shape[0]) - (1-alpha) * A_tilde)
    L = alpha * np.linalg.pinv(L_inv.toarray())
    return L

def Sum_Power_Tran(A):
    return

def Sim_Rank(A):
    return