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

class KL(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, score_type="basic", learning_rate=1e-2,weight_decay=1e-7,display_step=250):
        ''' Initialize the kl class

        Args:
            d: dimension of the embedding
            distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
        '''
        self._embedding_dim = embedding_dimension
        self._method_name = "KL"
        self._score_type = score_type
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._display_step = display_step
        self._epoch_begin = 0
        self._epoch_end = 0
        self._setup_done = False

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._embedding_dim}_{self._score_type}'

    def set_summary_writer(self, path):
        self._writer = SummaryWriter(path)

    def setup_model_input(self,AdjMat):
        self._num_nodes = AdjMat.shape[0]
        self._num_edges = AdjMat.sum()

        # Convert adjacency matrix to a CUDA Tensor
        self._adj = torch.FloatTensor(AdjMat.toarray()).cuda()
        
        #Calculate degree matrix from adjacency matrix
        self._degree=self._adj.sum(dim=1, dtype=torch.float)
        self._inv_degree=torch.diag(1/degree)
        self._degree=torch.diag(degree)

        # Define the embedding matrix
        self._emb = nn.Parameter(torch.empty(self._num_nodes, embedding_dim).normal_(0.0, 1.0))

        # Initialize the bias
        # The bias is initialized in such a way that if the dot product between two embedding vectors is 0 
        # (i.e. z_i^T z_j = 0), then their connection probability is sigmoid(b) equals to the 
        # background edge probability in the graph. This significantly speeds up training
        # TODO: WHY does it speed up the training?

        self._edge_proba = self._num_edges / (self._num_nodes**2 - self._num_nodes)
        self._bias_init = np.log(edge_proba / (1 - edge_proba))
        self._b = nn.Parameter(torch.Tensor([bias_init]))

        ### Optimizer definition ###
        # Regularize the embeddings but don't regularize the bias
        # The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        self._opt = torch.optim.Adam([
            {'params': [self._emb], 'weight_decay': self._weight_decay},
            {'params': [self._b]}],
            lr=self._learning_rate)
        self._setup_done = True

    def learn_embedding(self, num_epoch):

        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        self._epoch_end += num_epoch

        ## Define different loss functions

        def compute_loss_kl(adjmat_cuda, emb, degree):
            #KL(P||softmax(Z^TZ))=sum_i(P_i log (P_i / softmax(Z^TZ)_i))
            #As our objective is to minimize the KL divergence we can skip sum_i(P_i log(P_i)), which is const. in Z
            #This results in our objective to be min Z -sum_i(P_i log(softmax(Z^TZ)_i) )
            inv_degree=torch.diaglag(1/degree).cuda()
            P = inv_degree.mm(adjmat_cuda)
            loss = -(P*torch.log(10e-9+F.softmax(emb.mm(emb.t()),dim=1,dtype=torch.float)))
            return loss.mean()
            
        # other loss function
        
        # Choose loss function
        compute_loss = compute_loss_kl
        
        #### Model definition end ####
        
        #### Learning ####
        for epoch in range(self._epoch_begin, self._epoch_end):
            self._opt.zero_grad()
            loss = compute_loss(self._adj, self._emb)
            loss.backward()
            self._opt.step()
            # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._writer:
                # print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)

        # Put the embedding back on the CPU
        emb_np = self._emb.cpu().detach().numpy()

        # set epoch_begin to last epoch of training to ensure that loggin on tensorflow works correctly
        self._epoch_begin = self._epoch_end

        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np
