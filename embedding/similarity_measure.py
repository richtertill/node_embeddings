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

#helper:
def compute_degree_matrix(adj_np):
    degree= torch.from_numpy(adj_np.sum(axis=1)).cuda()
    return degree

# Adjacency Matrix A
def adjacency(adj_np):
    return torch.from_numpy(adj_np.todense()).cuda()

#Laplacian Matrix P=D-A
def laplacian( adj_np):
    adj_gpu=torch.FloatTensor(adj_np.toarray()).cuda()
    degree= compute_degree_matrix(adj_np)
    P = degree-adj_gpu
    return P

#Transition P=D^−1A
def compute_transition(adj_np):
    adj_gpu=torch.FloatTensor(adj_np.toarray()).cuda()
    degree= compute_degree_matrix(adj_np).float()
    inv_degree=torch.diagflat(1/degree)
    P = inv_degree.mm(adj_gpu) 
    return P


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
    return graph_util.csr_matrix_to_torch_tensor(L)

def NetMF(A):
    eps=1e-5
    #volume of the graph, usually for weighted graphs, here weight 1
    vol = A.sum()
    
    #b is the number of negative samples, hyperparameter
    b = 3
    
    #T is the window size, as a small window size algorithm is used, set T=10, which showed the best results in the paper
    T=10
    
    #Transition Matrix P=D^-1A
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    a=np.ones(D.shape[0])
    D_inv = np.divide(a, D, out=np.zeros_like(a), where=D!=0)
    P = np.diag(D_inv) * A.todense()
    
    #Compute M = vol(G)/bT (sum_r=1^T P^r)D^-1
    sum_np=0
    for r in range(1,T+1):
        sum_np+=np.linalg.matrix_power(P,r)
    M = sum_np * np.diag(D_inv) * vol / (b*T)
    M_max = np.maximum(M,np.ones(M.shape[0]))

    #Compute SVD of M
    u, s, vh = np.linalg.svd(np.log(M_max), full_matrices=True)

    #Compute L
    L = u*np.diag(np.sqrt(s+eps))
    # print(L.sum(axis=1))
    return graph_util.csr_matrix_to_torch_tensor(L)

#Symmetric, Normalized Laplacian P=D^(−1/2)AD^(−1/2)
def sym_normalized_laplacian_new(adj_np):
    adj_gpu=torch.FloatTensor(adj_np.toarray()).cuda()
    eps=1e-5
    degree= compute_degree_matrix(adj_np)
    sqrt = torch.sqrt(degree+eps).cuda()
    sqrtinv = torch.inverse(sqrt).cuda()
    P = torch.matmul(sqrtinv, torch.matmul(adj_gpu, sqrtinv)).cuda()
    return P

def NetMF_new(adj_gpu, adj_np):
    eps=1e-5
    #volume of the graph, usually for weighted graphs, here weight 1
    vol = A.sum()
    
    #b is the number of negative samples, hyperparameter
    b = 3
    
    #T is the window size, as a small window size algorithm is used, set T=10, which showed the best results in the paper
    T=10
    
    #Transition Matrix P=D^-1A
    P = compute_transition(adj_gpu, adj_np)
    
    #Compute M = vol(G)/bT (sum_r=1^T P^r)D^-1
    sum_torch=torch.zeros_like(P).cuda()
    for r in range(1,T+1):
        sum_torch+=P.pow(r)
    M = torch.matmul(sum_torch, torch.inverse(compute_degree_matrix(adj_np))).cuda() * vol / (b*T)
    M_max = torch.max(M,torch.ones(M.shape[0], M.shape[0])).cuda()

    #Compute SVD of M
    u, s, v = torch.svd(torch.log(M_max), some=False).cuda()

    #Compute L
    L = u*torch.diag(torch.sqrt(s+eps)).cuda()
    return L
    
#Personalized PageRank Matrix as described in https://openreview.net/pdf?id=H1gL-2A9Ym with the there used hyperparameter alpha=0.1
#P=alpha(I-(1-alpha)*D^-1/2(A+I)D^-1/2)^-1
def compute_ppr(adj_np, alpha = 0.1 ):
    adj_gpu=torch.FloatTensor(adj_np.toarray()).cuda()
    dim = adj_np.shape[0]
    term_2 = (1-alpha)*compute_transition(adj_np)
    term_1 = torch.eye(dim,device='cuda')
    matrix = term_1-term_2
    P = alpha*torch.inverse(matrix)
    return P


def compute_sum_power_tran(adj_np, T = 10 ):
    adj_gpu=torch.FloatTensor(adj_np.toarray()).cuda()
    matrix = compute_transition(adj_np)
    P = torch.zeros_like(matrix).cuda()
    for i in range(0,T+1):
        P = P + matrix.pow(i)
    return P

def simrank(adj_np, adj_gpu):
    #convert adjacency to networkx graph
    B = nx.from_scipy_sparse_matrix(adj_np)
    #built in simrank
    sim = nx.simrank_similarity(B)
    lol = [[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)]
    P = torch.tensor(lol)
    return P