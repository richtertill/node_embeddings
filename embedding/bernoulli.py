import os
import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from time import time
from .static_graph_embedding import StaticGraphEmbedding
from utils import graph_util
from .similarity_measure import adjacency


class Bernoulli(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, decoder='sigmoid', similarity_measure="adjacency",
                 learning_rate=1e-2, weight_decay=1e-7, display_step=100):
        '''
        Parameters
        ----------
        embedding_dimension
            Number of elements in the embedding vector representing a node, default 64.
        decoder
            One of {'sigmoid','guassian','exponential','dist2'}, default 'sigmoid'.
        sim_similarity_measure
            Only adjacency matrix is allowed as similarity measure, default 'adjacency'.
        learning_rate
            Initial learning rate for Adam optimizer, 1e-2.
        weight_decay
            Weight decay for Adam optimizer, default 1e-7.
        display_step
            Number of epochs after which the train error is logged for display on tensorboard and printed, default 25.
        '''
        self._embedding_dim = embedding_dimension
        self._decoder = decoder
        self._method_name = "Bernoulli"
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._display_step = display_step
        self._similarity_measure = similarity_measure
        self._epoch_begin = 0
        self._epoch_end = 0
        self._setup_done = False # model input is not setup yet 

    def setup_model_input(self, adj_mat):
        '''
        Parameters
        ----------
        adj_mat
            Adjacency matrix of the dataset to be tested in numpy sparse format.
        Return
        ------
        No explicit return value.
        But class variable self._Mat is assigned the similarity measure in the form of a torch tensor.
        '''

        self._Mat = adj_mat

        self._setup_done = True
        
    def get_method_name(self):
        '''        
        Return
        ------
        Name of embedding method as a string.
        '''
        return self._method_name

    def get_method_summary(self):
        '''        
        Return
        ------
        Name of entire model description including the name of the embedding method, the name of the decoder,
        the name of the used similarity measure and the number of embedding dimensions.
        '''
        return f'{self._method_name}_{self._decoder}_{self._similarity_measure}_{self._embedding_dim}'

    def reset_epoch(self):
        '''   
        This method resets start and end point of the training.   

        Return
        ------
        -
        '''
        self._epoch_begin = 0
        self._epoch_end = 0

    def get_embedding_dim(self):
        return self._embedding_dim

    def set_summary_folder(self, path):
        '''   
        This method creates a tensorboard summary writter which is used to log metrics during training and evaluation.
           
        Return
        ------
        -
        '''
        self._summary_path = path
        self._writer = SummaryWriter(self._summary_path)

    def get_summary_writer(self):
        '''   
        Return
        ------
        Reference to tensorboard summary writer (private class variable).
        '''
        return self._writer

    def learn_embedding(self, num_epoch):
        '''   
        This method to transform the input matrix of N x N down to N x embedding dimensions using the paramters specified during the initialization.
           
        Return
        ------
        Embedding matrix of N x embedding dimensions as numpy matrix on the CPU
        '''
        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        if self._epoch_begin != 0:
            self._epoch_begin = self._epoch_end +1
        else:
            self._epoch_begin = self._epoch_end
        self._epoch_end += num_epoch
        
        num_nodes = self._Mat.shape[0]
        num_edges = self._Mat.sum()
        
        embedding_dim = 64
        emb = nn.Parameter(torch.empty(num_nodes, embedding_dim).normal_(0.0, 1.0))

        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        bias_init = np.log(edge_proba / (1 - edge_proba))
        b = nn.Parameter(torch.Tensor([bias_init]))
        
        opt = torch.optim.Adam([
            {'params': [emb], 'weight_decay': 1e-7},
            {'params': [b]}],
        lr=1e-2)

        
        # Implementation of sigmoid decoder
        def compute_loss_sig(A, emb, b=0.0): 
            adj = torch.FloatTensor(A.toarray()).cuda()
            logits = emb @ emb.t() + b.cuda()
            loss = F.binary_cross_entropy_with_logits(logits, adj, reduction='none')
            loss[np.diag_indices(adj.shape[0])] = 0.0
            return loss.mean()
        
        # Implementation of guassian decoder
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
        
        # Implementation of "distance-2" decoder
        def compute_loss_dist2(A, emb, b=0.0):
            adj = torch.FloatTensor(A.toarray()).cuda()
            eps=1e-5
            pdist = ((emb[:, None] - emb[None, :]).pow(2.0).sum(-1) + eps).sqrt()
            logits = 10*(1-pdist).cuda()
            loss = F.binary_cross_entropy_with_logits(logits, adj, reduction='none')
            loss[np.diag_indices(adj.shape[0])] = 0.0
            return loss.mean()

        # Implementation of exponential decoder
        def compute_loss_exponential(adj, emb, b=0):
            eps=1e-5
            N = adj.shape[0]
            d=64
            e1, e2= adj.nonzero()
            emb_abs = torch.FloatTensor.abs(emb)
            dist = -torch.matmul(emb_abs,emb_abs.T)
            neg_term=dist
            neg_term[np.diag_indices(emb.shape[0])]=0.0
            expdist=torch.exp(dist)
            logdist=torch.log(1-expdist+eps)
            pos_term = logdist[e1,e2]
            return -(pos_term.sum() + neg_term.sum()) / emb.shape[0]**2

        if(self._decoder == "sigmoid"):
            compute_loss = compute_loss_sig
        if(self._decoder == "gaussian"):
            compute_loss = compute_loss_gaussian
        if(self._decoder == "exponential"):
            compute_loss = compute_loss_exponential
        if(self._decoder == "dist2"):
            compute_loss = compute_loss_dist2
        
        diff=  torch.FloatTensor([1e-7]).item()
        prev= torch.FloatTensor([1e-3]).item()

        for epoch in range(self._epoch_begin, self._epoch_end):
            opt.zero_grad()
            loss = compute_loss(self._Mat,emb.cuda(), b)
            loss.backward()
            opt.step()

            if epoch % self._display_step == 0 and self._summary_path:
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)
                
            now= loss.item()    
            if (abs(prev-now) < diff):
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                break
            else:
                prev=loss.item()
             
        # Put the embedding back on the CPU
        emb_np = emb.cpu().detach().numpy()

        return emb_np