'''
File for decoder functions:
Input: None
Initialize: Embedding Matrix
Output: Matrices for different decoders
'''

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from time import time

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from utils import graph_util

class decoder(AdjMat):

    def __init__(self, embedding_dimension=64):
        self._embedding_dim = embedding_dimension

    def setup_model_input(self, AdjMat):
        #input:
        self._num_nodes = AdjMat.shape[0]
        self._num_edges = AdjMat.sum()

        #setup learnable parts:
        self._emb = nn.Parameter(torch.empty(self._num_nodes, self._embedding_dim).normal_(0.0, 1.0))
        self._X = nn.Parameter(torch.empty(self._embedding_dim, self._embedding_dim).normal_(0.0, 1.0))
        self._edge_proba = self._num_edges / (self._num_nodes ** 2 - self._num_nodes)
        self._bias_init = np.log(self._edge_proba / (1 - self._edge_proba))
        self._b = nn.Parameter(torch.Tensor([self._bias_init]))

        ### Optimizer definition ###
        # Regularize the embeddings but don't regularize the bias
        # The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        self._opt = torch.optim.Adam([
            {'params': [self._emb], 'weight_decay': self._weight_decay},
            {'params': [self._b]}],
            lr=self._learning_rate)

        self._setup_done = True

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._embedding_dim}'

    def get_embedding_dim(self):
        return self._embedding_dim

    def get_summary_writer(self):
        return self._writer

    def decoder(self):
        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        def sigmoid(emb, b=0.1, eps=1e-5, similarity_measure):
            # embedding = sig(ZZ^T+b)
            e1, e2 = similarity_measure.nonzero()
            dist = torch.matmul(emb,emb.T) +b
            embedding = 1/(1+torch.exp(dist+eps)+eps)
            logsigdist = torch.log(sigdist+eps)
            pos_term = logsigdist[e1,e2]
            neg_term = torch.log(1-embedding)
            neg_term[np.diag_indices(emb.shape[0])] = 0.0
            size=emb.shape[0]
            return pos_term, neg_term, size, similarity_measure, embedding

        def sigmoidx(emb, X, b=0.1, eps=1e-5, similarity_measure):
            # embedding = sig(ZXZ^T+b)
            e1, e2 = similarity_measure.nonzero()
            dist = torch.matmul(emb,torch.matmul(X,emb.T)) +b
            embedding = 1/(1+torch.exp(dist+eps)+eps)
            logsigdist = torch.log(embedding+eps)
            pos_term = logsigdist[e1,e2]
            neg_term = torch.log(1-embedding)
            neg_term[np.diag_indices(emb.shape[0])] = 0.0
            size=emb.shape[0]
            return pos_term, neg_term, size, embedding, similarity_measure, embedding

        def distance(emb, eps=1e-5, similarity_measure):
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
            return pos_term, neg_term, size, similarity_measure, embedding

        def exponential(emb, eps=1e-5, similarity_measure):
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
            return pos_term, neg_term, size, similarity_measure, embedding

        def exponentialx(emb, eps=1e-5, similarity_measure, X):
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
            return pos_term, neg_term, size, similarity_measure, embedding

        # Choose decoder
        if self._decoder == 'sigmoid':
            if self.X == True:
                decoder = sigmoidx
            else:
                decoder = sigmoid
        elif self._decoder == 'distance':
            decoder = distance
        elif self._decoder == 'exponential':
            if self.X == True:
                decoder = exponentialx
            else:
                decoder = exponential