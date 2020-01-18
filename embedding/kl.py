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
# from .similarity_measure import adjacency, laplacian, sym_normalized_laplacian, NetMF, ppr, sum_power_tran, sim_rank

class KL(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, decoder='sigmoid', similarity_measure="adjacency", W_enabled=False,
                 learning_rate=1e-2, weight_decay=1e-7, display_step=25):
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
        self._W_enabled = W_enabled

    def setup_model_input(self, adj_mat):
        # input
        self._num_nodes = adj_mat.shape[0]
        self._num_edges = adj_mat.sum()
        self._Adj = adj_mat

        # Model parameters
       
        self._setup_done = True

    def get_similarity_measure(self):
        '''
        return given similarity measure here and particularly e1 and e2, we need to find an elegang way for this
        '''
        return self._similarity_measure
        
    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._embedding_dim}_{self._decoder}_{self._similarity_measure}_{self._W_enabled}'

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

# Regularize the embeddings but don't regularize the bias
# The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        opt = torch.optim.Adam([
            {'params': [emb], 'weight_decay': 1e-7},
            {'params': [b]}],
        lr=1e-2)
        
        
        def compute_degree_matrix(adj_np):
            degree= torch.from_numpy(adj_np.sum(axis=1)).cuda()
            return degree
    
        def compute_transition(adj_gpu,adj_np):
            degree= compute_degree_matrix(adj_np).float()
            inv_degree=torch.diagflat(1/degree)
            P = inv_degree.mm(adj_gpu) 
            return P

        def compute_ppr(adj_gpu, adj_np, dim, alpha = 0.1 ):
            term_2 = (1-alpha)*compute_transition(adj_gpu,adj_np)
            term_1 = torch.eye(dim,device='cuda')
            matrix = term_1-term_2
            P = alpha*torch.inverse(matrix)
            return P


        def compute_spt(adj_gpu, adj_np, T = 10 ):
            matrix = compute_transition(adj_gpu,adj_np)
            P = torch.zeros_like(matrix).cuda()
            for i in range(1,5):
                P = P + matrix.pow(i)
            return P

        def compute_loss_KL(adj_np, emb, method, b=0.0):
            N,D=adj_np.shape
            adj_gpu = torch.FloatTensor(adj_np.toarray()).cuda()
            if method=='transition':
                sim = compute_transition(adj_gpu,adj_np)
            elif method=='ppr':
                sim = compute_ppr(adj_gpu,adj_np,N)
            elif method=='sum_power_tran':
                sim = compute_spt(adj_gpu,adj_np)
            else:
                print("The similarity measure does not exist")
            loss = -(sim*torch.log( 10e-9+ F.softmax(emb.mm(emb.t()),dim=1,dtype=torch.float)))
            return loss.mean()
        
        compute_loss = compute_loss_KL

        for epoch in range(self._epoch_begin, self._epoch_end):
            opt.zero_grad()
            loss = compute_loss(A, emb.cuda(),self._similarity_measure, b)
            loss.backward()
            opt.step()
    # Training loss is printed every display_step epochs
            if epoch % self._display_step == 0 and self._summary_path:
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
                self._writer.add_scalar('Loss/train', loss.item(), epoch)
        

   
        # Put the embedding back on the CPU
        emb_np = emb.cpu().detach().numpy()

        return emb_np