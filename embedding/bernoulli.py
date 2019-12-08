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


class Bernoulli(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, distance_meassure='sigmoid',
                 learning_rate=1e-2, weight_decay=1e-7, display_step=250):
        ''' Initialize the Bernoulli class

        Args:
            d: dimension of the embedding
            distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
        '''
        self._embedding_dim = embedding_dimension
        self._distance_meassure = distance_meassure
        self._method_name = "Bernoulli"
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._display_step = display_step
        self._epoch_begin = 0
        self._epoch_end = 0
        self._setup_done = False

    def setup_model_parameters(self, AdjMat):
        # input
        self._num_nodes = AdjMat.shape[0]
        self._num_edges = AdjMat.sum()

        # Model parameters
        self._emb = nn.Parameter(torch.empty(self._num_nodes, self._embedding_dim).normal_(0.0, 1.0))
        self._edge_proba = self._num_edges / (self._num_nodes ** 2 - self._num_nodes)
        self._bias_init = np.log(self._edge_proba / (1 - self._edge_proba))
        self._b = nn.Parameter(torch.Tensor([self._bias_init]))

        self._e1,self._e2 = AdjMat.nonzero()
        
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
        return '%s_%d' % (self._method_name, self._embedding_dim)

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

        # sigmoid loss function
        def compute_loss_ber_sig(emb, b=0.1, eps=1e-5):
            dist = torch.matmul(emb,emb.T) +b
            sigdist = 1/(1+torch.exp(dist+eps)+eps)
            logsigdist = torch.log(sigdist+eps)
            pos_term = logsigdist[self._e1,self._e2]
            neg_term = torch.log(1-sigdist)
            neg_term[np.diag_indices(N)] = 0.0

            return -(pos_term.sum() + neg_term.sum()) / emb.shape[0]**2

        def compute_loss_ber_dist(emb, eps=1e-5):
            pdist = ((emb[:, None] - Z[None, :]).pow(2.0).sum(-1) + eps).sqrt()
            neg_term = torch.log(-torch.expm1(-pdist) + 1e-5)
            neg_term[np.diag_indices(N)] = 0.0
            pos_term = -pdist[self._e1, self._e2]
            neg_term[self._e1, self._e2] = 0.0
            return -(pos_term.sum() + neg_term.sum()) / emb.shape[0] ** 2

        def compute_loss_ber_exp(emb, eps=1e-5):
            emb_abs = torch.FloatTensor.abs(Z)
            dist = -torch.matmul(emb_abs,emb_abs.T)
            neg_term=dist
            neg_term[np.diag_indices(N)]=0.0
            expdist=torch.exp(dist)
            logdist=torch.log(1-expdist+eps)
            pos_term = logdist[self._e1,self._e2]

            return -(pos_term.sum() + neg_term.sum()) / emb.shape[0]**2

        # Choose loss function
        if self._distance_meassure == 'sigmoid':
            compute_loss = compute_loss_ber_sig
        elif self._distance_meassure == 'distance':
            compute_loss = compute_loss_ber_dist
        elif self._distance_meassure == 'exponential':
            compute_loss = compute_loss_ber_exp

        #### Learning ####

        # Training loop
        for epoch in range(self._epoch_begin, self._epoch_end):
            self._opt.zero_grad()
            loss = compute_loss(self._adj, self._emb)
            loss.backward()
            self._opt.step()
            # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._summary_path:
                # print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)

        # Put the embedding back on the CPU
        emb_np = self._emb.cpu().detach().numpy()

        # set epoch_begin to last epoch of training to ensure that loggin on tensorflow works correctly
        self._epoch_begin = self._epoch_end

        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np