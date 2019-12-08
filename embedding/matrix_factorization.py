import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from time import time
from scipy.sparse import coo_matrix

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from utils import graph_util


class MatrixFactorization(StaticGraphEmbedding):

    def __init__(self, embedding_dimension=64, matrix="adjacency"):
        ''' Initialize the Matrix Factorization class

        Args:

        '''
        self._embedding_dim = embedding_dimension
        self._matrix = matrix
        self._method_name = "Matrix_Fatorization"
        self._setup_done = False

    def setup_model_parameters(self, AdjMat):

        if (self._matrix=="adjacency"):
            coo = coo_matrix(AdjMat)
            values = coo.data
            indices = np.vstack((coo.row, coo.col))

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape
            self._AdjMat = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
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

    def learn_embedding(self, num_epochs):

        if self._setup_done == False:
            raise ValueError('Model input parameters not defined.')

        #### Learning ####
        U,S,V = torch.svd(self._AdjMat)

        self._emb = U[:,:self._embedding_dim]

        # Put the embedding back on the CPU
        emb_np = self._emb.cpu().detach().numpy()


        # Save the embedding
        #         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)

        return emb_np