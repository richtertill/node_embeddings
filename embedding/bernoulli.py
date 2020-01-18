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
# from .decoder import sigmoid, gaussian, exponential
from .similarity_measure import adjacency



class Bernoulli(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, decoder='sigmoid', W_enabled=False,
                 learning_rate=1e-2, weight_decay=1e-7, display_step=25):
        ''' Initialize the Bernoulli class

        Args:
            d: dimension of the embedding
            decoder: name of decoder ('sigmoid','gaussian',...)
        '''
        self._embedding_dim = embedding_dimension
        self._decoder = decoder
        self._method_name = "Bernoulli"
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._display_step = display_step
        self._epoch_begin = 0
        self._epoch_end = 0
        self._setup_done = False
        self._similarity_measure = "adjacency"
        self._W_enabled = W_enabled

    def setup_model_input(self, adj_mat):
        # input
       
        self._num_nodes = adj_mat.shape[0]
        self._num_edges = adj_mat.sum()
        self._Adj = adj_mat

        self._setup_done = True

    def get_similarity_measure(self):
        '''
        return given similarity measure here and particularly e1 and e2, we need to find an elegang way for this
        '''
        return self._similarity_measure
        
    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._decoder}_{self._similarity_measure}_{self._embedding_dim}_{self._W_enabled}'

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

        if self._epoch_begin != 0:
            self._epoch_begin = self._epoch_end +1
        else:
            self._epoch_begin = self._epoch_end
        self._epoch_end += num_epoch
        
        A = self._Adj
        num_nodes = A.shape[0]
        num_edges = A.sum()
        
        embedding_dim = 64
        emb = nn.Parameter(torch.empty(num_nodes, embedding_dim).normal_(0.0, 1.0))

        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        bias_init = np.log(edge_proba / (1 - edge_proba))
        b = nn.Parameter(torch.Tensor([bias_init]))
        
        opt = torch.optim.Adam([
            {'params': [emb], 'weight_decay': 1e-7},
            {'params': [b]}],
        lr=1e-2)

        
      
        def compute_loss_sig(A, emb, b=0.0): 
            adj = torch.FloatTensor(A.toarray()).cuda()
            logits = emb @ emb.t() + b.cuda()
            loss = F.binary_cross_entropy_with_logits(logits, adj, reduction='none')
            loss[np.diag_indices(adj.shape[0])] = 0.0
            return loss.mean()
        
        
        def compute_loss_gaussian(adj, emb, b=0.0):
            eps=1e-5
            N = adj.shape[0]
            d=64
            e1, e2= adj.nonzero()
            pdist = ((emb[:, None] - emb[None, :]).pow(2.0).sum(-1) + eps).sqrt()
            neg_term = torch.log(-torch.expm1(-pdist) + 1e-5)
            neg_term[np.diag_indices(N)] = 0.0
            pos_term = -pdist[e1, e2]
            neg_term[e1, e2] = 0.0
            return -(pos_term.sum() + neg_term.sum()) / emb.shape[0]**2




        if(self._decoder == "sigmoid"):
            compute_loss = compute_loss_sig
        if(self._decoder == "gaussian"):
            compute_loss = compute_loss_gaussian
        if(self._decoder == "exponential"):
            compute_loss = compute_loss_exponential
        
        
        
        for epoch in range(self._epoch_begin, self._epoch_end):
            opt.zero_grad()
            loss = compute_loss(A ,emb.cuda(), b)
            loss.backward()
            opt.step()
    # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._summary_path:
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)
        

   
        # Put the embedding back on the CPU
        emb_np = emb.cpu().detach().numpy()

        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np