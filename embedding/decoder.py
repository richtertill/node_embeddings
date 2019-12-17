'''
File for decoder functions:
Input: None
Initialize: Embedding Matrix
Output: Matrices for different decoders
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from time import time

from .similarity_measure import adjacency, laplacian, dw

def sigmoid(emb,similarity_measure, b=0.1, eps=1e-5):
    # embedding = sig(ZZ^T+b)
    e1, e2 = similarity_measure.nonzero()
    dist = torch.matmul(emb,emb.T) +b
    embedding = 1/(1+torch.exp(dist+eps)+eps)
    logsigdist = torch.log(sigdist+eps)
    pos_term = logsigdist[e1,e2]
    neg_term = torch.log(1-embedding)
    neg_term[np.diag_indices(emb.shape[0])] = 0.0
    size=emb.shape[0]
    return pos_term, neg_term, size, embedding

def sigmoidx(emb, X,similarity_measure, b=0.1, eps=1e-5):
    # embedding = sig(ZXZ^T+b)
    e1, e2 = similarity_measure.nonzero()
    dist = torch.matmul(emb,torch.matmul(X,emb.T)) +b
    embedding = 1/(1+torch.exp(dist+eps)+eps)
    logsigdist = torch.log(embedding+eps)
    pos_term = logsigdist[e1,e2]
    neg_term = torch.log(1-embedding)
    neg_term[np.diag_indices(emb.shape[0])] = 0.0
    size=emb.shape[0]
    return pos_term, neg_term, size, embedding, embedding

def gaussian(emb, similarity_measure, eps=1e-5,):
    # embedding = exp(-gamma||z_i-z_j||^2)
    gamma = 0.1
    e1, e2 = similarity_measure.nonzero()
    pdist = ((emb[:, None] - emb[None, :]).pow(2.0).sum(-1) + eps).sqrt()
    embedding = torch.expm1(-pdist*gamma) + 1e-5
    neg_term = torch.log(embedding)
    neg_term[np.diag_indices(emb.shape[0])] = 0.0
    pos_term = -pdist[e1, e2]
    neg_term[e1, e2] = 0.0
    size=emb.shape[0]
    return pos_term, neg_term, size, embedding

def exponential(emb, similarity_measure, eps=1e-5):
    # embedding = 1 - exp(-ZZ^T)
    e1, e2 = similarity_measure.nonzero()
    emb_abs = torch.FloatTensor.abs(emb)
    dist = -torch.matmul(emb_abs, emb_abs.T)
    neg_term = dist
    neg_term[np.diag_indices(emb.shape[0])] = 0.0
    expdist = torch.exp(dist)
    embedding = 1 - expdist
    logdist = torch.log(embedding + eps)
    pos_term = logdist[e1, e2]
    size=emb.shape[0]
    return pos_term, neg_term, size, embedding

def exponentialx(emb, similarity_measure, X, eps=1e-5):
    # embedding = 1 - exp(-ZXZ^T)
    e1, e2 = similarity_measure.nonzero()
    emb_abs = torch.FloatTensor.abs(emb)
    x_abs = torch.FloatTensor.abs(X)
    dist = -torch.matmul(emb_abs, torch.matmul(x_abs,emb_abs.T))
    neg_term = dist
    neg_term[np.diag_indices(emb.shape[0])] = 0.0
    expdist = torch.exp(dist)
    embedding = 1 - expdist
    logdist = torch.log(embedding + eps)
    pos_term = logdist[e1, e2]
    size=emb.shape[0]
    return pos_term, neg_term, size, embedding