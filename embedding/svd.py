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
from gust import preprocessing as GustPreprosessing


class MatrixFactorization(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, matrix_type="adjacency"):
        '''
        Parameters
        ----------
        embedding_dimension
            Number of elements in the embedding vector representing a node.
        matrix_type
            One of {'adjacency','unnormalized_lapla', 'random_walk_lapla', 'symmetrized_lapla'}, default 'adjacency'.
            Type of the Laplacian to compute.

            adjacency = A
            unnormalized_lapla = D - A
            random_walk_lapla = I - D^{-1} A
            symmetrized_lapla = I - D^{-1/2} A D^{-1/2}

        Returns
        -------
        sp.csr_matrix
            Laplacian matrix in the same format as A.
        '''
        self._embedding_dim = embedding_dimension
        self._matrix_type = matrix_type
        self._method_name = "Matrix_Fatorization"
        self._setup_done = False

    def setup_model_input(self, adj_mat, matrix_type=None):

        if(matrix_type):
            self._matrix_type = matrix_type

        # transform matrix to correct type
        if (self._matrix_type=="adjacency"):
            matrix = adj_mat 
        if (self._matrix_type=="laplacian"):
            matrix = GustPreprosessing.construct_laplacian(adj_mat,type="unnormalized")
        if (self._matrix_type=="random_walk_lapla"):
            matrix = GustPreprosessing.construct_laplacian(adj_mat,type="random_walk")
        if (self._matrix_type=="symmetrized_lapla"):
            matrix = GustPreprosessing.construct_laplacian(adj_mat,type="symmetrized")
                    
        # sparse matrix to cuda tensor
        self._Mat = graph_util.csr_matrix_to_torch_tensor(matrix)
        self._setup_done = True

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return f'{self._method_name}_{self._embedding_dim}_{self._matrix_type}'

    def reset_epoch(self):
        self._epoch_begin = 0
        self._epoch_end = 0

    def set_summary_folder(self, path):
        self._summary_path = path
        self._writer = SummaryWriter(self._summary_path)

    def get_summary_writer(self):
        return self._writer

    def learn_embedding(self, num_epochs):

        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        #### Learning ####
        U,S,V = torch.svd(self._Mat)
        self._emb = U[:,:self._embedding_dim]

        # Put the embedding back on the CPU
        emb_np = self._emb.cpu().detach().numpy()


        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np