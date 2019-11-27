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

    def __init__(self, embedding_dimension=64, distance_meassure='sigmoid', max_epoch=5000, learning_rate=1e-2,
                 weight_decay=1e-7, display_step=250):
        ''' Initialize the Bernoulli class

        Args:
            d: dimension of the embedding
            distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
        '''

        self._d = embedding_dimension
        self._distance_meassure = distance_meassure
        self._method_name = "Bernoulli"
        self._max_epoch = max_epoch
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._display_step = display_step

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def get_embedding_dim(self):
        return _d

    def set_summary_folder(self, path):
        self._summary_path = path
        self._writer = SummaryWriter(self._summary_path)

    def learn_embedding(self, AdjMat):

        self._num_nodes = AdjMat.shape[0]
        self._num_edges = AdjMat.sum()

        # Convert adjacency matrix to a CUDA Tensor
        adj = torch.FloatTensor(AdjMat.toarray()).cuda()
        #### Model definition ####

        # Define the embedding matrix
        embedding_dim = 64
        emb = nn.Parameter(torch.empty(self._num_nodes, embedding_dim).normal_(0.0, 1.0))

        # Initialize the bias
        # The bias is initialized in such a way that if the dot product between two embedding vectors is 0
        # (i.e. z_i^T z_j = 0), then their connection probability is sigmoid(b) equals to the
        # background edge probability in the graph. This significantly speeds up training
        # TODO: WHY does it speed up the training?
        edge_proba = self._num_edges / (self._num_nodes ** 2 - self._num_nodes)
        bias_init = np.log(edge_proba / (1 - edge_proba))
        b = nn.Parameter(torch.Tensor([bias_init]))

        ## Define different loss functions

        # sigmoid loss function
        def compute_loss_ber_sig(adj, emb, b=0.1):
            """Compute the NLL of Bernoulli using a Sigmoid Kernel"""
            N, d = emb.shape

            # compute f(z_i, z_j) = sigma(z_i^Tz_j+b)
            dot = torch.matmul(emb, emb.T)
            logits = dot + b

            # transform adj
            ind = torch.triu_indices(N, N, offset=1)
            logits = logits[ind[0], ind[1]]
            labels = adj[ind[0], ind[1]]

            # compute p(A|Z)
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=None, size_average=None, reduce=None,
                                                      reduction='mean')
            return loss

        def compute_loss_ber_exp1(adj, emb, b=0.0):
            """Compute the rdf distance of the Bernoulli model."""
            """Currently very slow, as fast way has a bug"""
            # Initialization
            N, d = emb.shape
            squared_euclidian = torch.zeros(N, N).cuda()
            gamma = 0.1
            # Compute squared euclidian
            for index, embedding in enumerate(emb):
                sub = embedding - emb + 10e-9
                squared_euclidian[index, :] = torch.sum(torch.pow(sub, 2), 1)
            # Compute exponentianl
            radial_exp = torch.exp(-gamma * torch.sqrt(squared_euclidian))
            loss = F.binary_cross_entropy(radial_exp, adj, reduction='none')
            loss[np.diag_indices(adj.shape[0])] = 0.0
            return loss.mean()


        def compute_loss_ber_exp2(adj, emb):
            """Compute the NLL of Bernoulli using an Exponential Kernel"""
            N, d = emb.shape

            # get indices of upper triangular matrix
            ind = torch.triu_indices(N, N, offset=1)

            # compute f(z_i, z_j) = sigma(z_i^Tz_j+b)
            dot = torch.matmul(emb, emb.T)
            logits = 1 - torch.exp(-dot)
            logits = logits[ind[0], ind[1]]
            labels = adj[ind[0], ind[1]]

            # compute loss
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

            return loss



        # Choose loss function
        if self._distance_meassure == 'sigmoid':
            compute_loss = compute_loss_sigmoid
        elif self._distance_meassure == 'distance':
            compute_loss = compute_loss_ber_exp1
        elif self._distance_meassure == 'exponential':
            compute_loss = compute_loss_ber_exp2

        #### Model definition end ####

        #### Learning ####

        # Regularize the embeddings but don't regularize the bias
        # The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        opt = torch.optim.Adam([
            {'params': [emb], 'weight_decay': self._weight_decay},
            {'params': [b]}],
            lr=self._learning_rate)

        # Training loop
        for epoch in range(self._max_epoch):
            opt.zero_grad()
            loss = compute_loss(adjmat_cuda, emb, b)
            loss.backward()
            opt.step()
            # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._summary_path:
                # print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)

        # Put the embedding back on the CPU
        emb_np = emb.cpu().detach().numpy()

        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np

#     def get_embedding(self, filesuffix=None):
#         return self._Y if filesuffix is None else np.loadtxt(
#             'embedding_' + filesuffix + '.txt'
#         )
