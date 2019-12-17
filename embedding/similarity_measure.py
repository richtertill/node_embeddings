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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from time import time

def adjacency(A):
    # Adjacency Matrix A
    return torch.FloatTensor(A.toarray()).cuda()

def laplacian(A):
    #Laplacian P=D^−1A
    degreenp=A.sum(axis=1)
    degree = torch.from_numpy(degreenp).cuda().view(-1)
    degreematrix = torch.zeros(N,N)
    degreematrix[np.diag_indices(N)]=degree
    invdegree = torch.inverse(degreematrix)
    adj = torch.FloatTensor(A.toarray()).cuda()
    return torch.matmul(invdegree, adj)

def dw(A):
    #Deep Walk / Symmetric, Normalized Laplacian D^(−1/2)AD^(−1/2)
    degreenp=A.sum(axis=1)
    degree = torch.from_numpy(degreenp).cuda().view(-1)
    degreematrix = torch.zeros(N,N)
    degreematrix[np.diag_indices(N)]=degree
    sqrtdegree = degreematrix.sqrt()
    invdegree = torch.inverse(sqrtdegree)
    adj = torch.FloatTensor(A.toarray()).cuda() 
    return torch.matmul(invdegree, torch.matmul(adj, invdegree))