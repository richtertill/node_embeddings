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

from .decoder import sigmoid, gaussian, exponential
from .similarity_measure import adjacency, laplacian, transition, sym_normalized_laplacian, NetMF, ppr, sum_power_tran, sim_rank

class KL(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, decoder='sigmoid', similarity_measure="adjacency",
                 learning_rate=1e-2, weight_decay=1e-7, display_step=10):
        ''' Initialize the Bernoulli class

        Args:
            d: dimension of the embedding
            distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
        '''
        self._embedding_dim = embedding_dimension
        self._decoder = decoder
        self._method_name = "KL"
        self._learning_rate = learning_rate
        self._similarity_measure = similarity_measure
        self._weight_decay = weight_decay
        self._display_step = display_step
        self._epoch_begin = 0
        self._epoch_end = 0
        self._setup_done = False

    def setup_model_input(self, adj_mat):
        # input
        self._num_nodes = adj_mat.shape[0]
        self._num_edges = adj_mat.sum()

        # Model parameters
        self._emb = nn.Parameter(torch.empty(self._num_nodes, self._embedding_dim).normal_(0.0, 0.1))
        self._X = nn.Parameter(torch.empty(self._num_nodes, self._embedding_dim).normal_(0.0, 1.0))
        self._edge_proba = self._num_edges / (self._num_nodes ** 2 - self._num_nodes)
        self._bias_init = np.log(self._edge_proba / (1 - self._edge_proba))
        self._b = nn.Parameter(torch.Tensor([self._bias_init]))


        if (self._similarity_measure=="transition"):
            self._Mat = transition(adj_mat)
        if (self._similarity_measure=="NetMF"):
            self._Mat = NetMF(adj_mat)
        if (self._similarity_measure=="ppr"):
            self._Mat = ppr(adj_mat)
        if (self._similarity_measure=="sum_power_tran"):
            self._Mat = sum_power_tran(adj_mat)

        self._Mat = self._Mat
        
        ### Optimizer definition ###
        # Regularize the embeddings but don't regularize the bias
        # The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        self._opt = torch.optim.Adam([
            {'params': [self._emb], 'weight_decay': self._weight_decay},
            {'params': [self._b]}],
            lr=self._learning_rate)

        self._setup_done = True

    def get_similarity_measure(self):
        '''
        return given similarity measure here and particularly e1 and e2, we need to find an elegang way for this
        '''
        return self._similarity_measure
        
    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._embedding_dim}_{self._decoder}_{self._similarity_measure}'

    def reset_epoch(self):
        self._epoch_begin = 0
        self._epoch_end = 0

    def get_embedding_dim(self):
        return self._embedding_dim

    def set_summary_folder(self, path):
        self._summary_path = path
        self._writer = SummaryWriter(self._summary_path)

    def get_summary_writer(self):
        return self._writer

    def learn_embedding(self, num_epoch):

        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        self._epoch_end += num_epoch
            
        def compute_loss_softmax(emb, b=0.1, eps=1e-5):
            dist = torch.matmul(emb, emb.T)+b
            softmax = nn.Softmax(dim=0)
            embedding = softmax(dist)
            #mat = softmax(self._Mat)
            #embedding = nn.Softmax(emb, dim=0)
            # embedding = embedding.to(torch.device("cuda"))
            return -(torch.matmul(self._Mat, torch.log(embedding + eps))).sum()
            #return -(torch.matmul(mat, torch.log(embedding + eps))).sum()

        #### Learning ####

        # Training loop
        for epoch in range(self._epoch_begin, self._epoch_end+1):
            self._opt.zero_grad()
            loss = compute_loss_softmax(self._emb)
            loss.backward()
            self._opt.step()
            # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._summary_path:
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)

        # Put the embedding back on the CPU
        emb_np = self._emb.cpu().detach().numpy()

        # set epoch_begin to last epoch of training to ensure that loggin on tensorflow works correctly
        self._epoch_begin = self._epoch_end

        return emb_np