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
    return graph_util.csr_matrix_to_torch_tensor(L)

def transition(A):
    #Laplacian P=D^−1A
    num_nodes = A.shape[0]
    D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    a=np.ones(D.shape[0])
    D_inv = np.divide(a, D, out=np.zeros_like(a), where=D!=0)
    L = sp.diags(D_inv) * A
    return graph_util.csr_matrix_to_torch_tensor(L)

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

def ppr(A):
    #Personalized PageRank Matrix as described in https://openreview.net/pdf?id=H1gL-2A9Ym with the there used hyperparameter alpha=0.1
    #P=alpha(I-(1-alpha)*D^-1/2(A+I)D^-1/2)^-1
    alpha = 0.1  
    #num_nodes = A.shape[0]
    #D = np.ravel(A.sum(1))
    #D[D == 0] = 1  # avoid division by 0 error
    #D_sqrt = np.sqrt(D)
    #a=np.ones(D_sqrt.shape[0])
    #D_sqrt_inv = np.divide(a, D_sqrt, out=np.zeros_like(a), where=D!=0)
    #A_tilde = sp.diags(D_sqrt_inv) * (A + sp.identity(A.shape[0])) * sp.diags(D_sqrt_inv)
    A = graph_util.csr_matrix_to_torch_tensor(A)
    A_tilde = torch.tensor(A / torch.sum(A, 1,keepdims=True))
    L = alpha*torch.inverse(torch.eye(A.shape[0])-(1-alpha)*A_tilde)
    #L_inv = (sp.identity(A.shape[0]) - (1-alpha) * A_tilde)
    #L = alpha * np.linalg.pinv(L_inv.toarray())
    #return graph_util.csr_matrix_to_torch_tensor(L)
    return L

def sum_power_tran(A):
    #T is the window size, as a small window size algorithm is used, set T=10, which showed the best results in the paper
    T=10
    
    #volume of the graph, usually for weighted graphs, here weight 1
    vol = A.sum()
    #b is the number of negative samples, hyperparameter
    b = 3

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
    M_max = np.maximum(M,np.ones(M.shape[0])) #this step is proposed to yield stability, if eg log is applied
    return graph_util.csr_matrix_to_torch_tensor(M_max)

def sim_rank(A, C = 0.8, acc = 0.1):
    #https://link.springer.com/chapter/10.1007/978-3-642-14246-8_29
    #Algorithm 1: AUG-SimRank: Accelerative SimRank for Undirected Graphs
    A = torch.tensor(A.todense())
    
    #Calculate Transition Probability Q
    Q = A / A.sum(1, keepdims=True)
    
    #Decompose Q
    eigvalues, eigvectors = torch.eig(Q, eigenvectors=True)
    #for undirected graphs all eigenvalues are real
    eigvalues = eigvalues[:,0]
    
    #Initialize
    S_old = torch.eye(Q.shape[0])
    M = C * torch.diag(eigvalues) @ torch.diag(eigvalues).T
    #k=0
    
    #Converge
    while True:
        #k+=1
        S_new = torch.max(M*S_old.float(),torch.eye(M.shape[0]).double())
        
        if torch.max(torch.abs(S_new-S_old))<acc:
            break
        S_old = S_new
    
    L = eigvectors @ S_new @ torch.inverse(eigvectors)
    
    return L