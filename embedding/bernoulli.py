disp_avlbl = True
import os
if 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import scipy.io as sio
import pdb

import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from subprocess import call

from .static_graph_embedding import StaticGraphEmbedding


import gust  # library for loading graph data
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

torch.set_default_tensor_type('torch.cuda.FloatTensor')
sns.set_style('whitegrid')


from gem.utils import graph_util, plot_util
from gem.evaluation import visualize_embedding as viz
from time import time


class Bernoulli(StaticGraphEmbedding):

    """`Bernoulli`_.
    Bernoulli based embedding assumes every element in the embedding vector follows a bernoulli distribution.
    
    Args:
        hyper_dict (object): Hyper parameters.
        kwargs (dict): keyword arguments, form updating the parameters
    
    Examples:
        >>> from gemben.embedding.gf import GraphFactorization
        >>> edge_f = 'data/karate.edgelist'
        >>> G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
        >>> G = G.to_directed()
        >>> res_pre = 'results/testKarate'
        >>> graph_util.print_graph_stats(G)
        >>> t1 = time()
        >>> embedding = GraphFactorization(2, 100000, 1 * 10**-4, 1.0)
        >>> embedding.learn_embedding(graph=G, edge_f=None,
                                  is_weighted=True, no_python=True)
        >>> print ('Graph Factorization:Training time: %f' % (time() - t1))
        >>> viz.plot_embedding2D(embedding.get_embedding(),
                             di_graph=G, node_colors=None)
        >>> plt.show()
    .. _Graph Factorization:
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40839.pdf
    """

    def __init__(self, *hyper_dict, **kwargs):
        ''' Initialize the GraphFactorization class
        Args:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        '''
        hyper_params = {
            'print_step': 10000,
            'method_name': 'bernoulli'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def _get_f_value(self, graph):
        f1 = 0
        for i, j, w in graph.edges(data='weight', default=1):
            f1 += (w - np.dot(self._X[i, :], self._X[j, :]))**2
        f2 = self._regu * (np.linalg.norm(self._X)**2)
        return [f1, f2, f1 + f2]

    def learn_embedding(self, A, embedding_dim=64): # A: adjacency matrix
        
        num_nodes = A.shape[0]
        num_edges = A.sum()

        # Convert adjacency matrix to a CUDA Tensor
        adj = torch.FloatTensor(A.toarray()).cuda()
        emb = nn.Parameter(torch.empty(num_nodes, embedding_dim).normal_(0.0, 1.0))

        # Initialize the bias
        # The bias is initialized in such a way that if the dot product between two embedding vectors is 0 
        # (i.e. z_i^T z_j = 0), then their connection probability is sigmoid(b) equals to the 
        # background edge probability in the graph. This significantly speeds up training
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        bias_init = np.log(edge_proba / (1 - edge_proba))
        b = nn.Parameter(torch.Tensor([bias_init]))

        # Regularize the embeddings but don't regularize the bias
        # The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
        opt = torch.optim.Adam([
            {'params': [emb], 'weight_decay': 1e-7},
            {'params': [b]}],
            lr=1e-2)
        
        def compute_loss_v1(adj, emb, b=0.0): 
            """Compute the negative log-likelihood of the Bernoulli model."""
            logits = emb @ emb.t() + b
            loss = F.binary_cross_entropy_with_logits(logits, adj, reduction='none')
            # Since we consider graphs without self-loops, we don't want to compute loss
            # for the diagonal entries of the adjacency matrix.
            # This will kill the gradients on the diagonal.
            loss[np.diag_indices(adj.shape[0])] = 0.0
            return loss.mean()
        
        
        max_epochs = 5000
        display_step = 250

        for epoch in range(max_epochs):
            opt.zero_grad()
            loss = compute_loss(adj, emb, b)
            loss.backward()
            opt.step()
            # Training loss is printed every display_step epochs
            if epoch % display_step == 0:
                print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
        
        
        
        
        c_flag = True
        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if no_python:
            if sys.platform[0] == "w":
                args = ["gem/c_exe/gf.exe"]
            else:
                args = ["gem/c_exe/gf"]
            if not graph and not edge_f:
                raise Exception('graph/edge_f needed')
            if edge_f:
                graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
            graphFileName = 'gem/intermediate/%s_gf.graph' % self._data_set
            embFileName = 'gem/intermediate/%s_%d_gf.emb' % (self._data_set, self._d)
            # try:
                # f = open(graphFileName, 'r')
                # f.close()
            # except IOError:
            graph_util.saveGraphToEdgeListTxt(graph, graphFileName)
            args.append(graphFileName)
            args.append(embFileName)
            args.append("1")  # Verbose
            args.append("1")  # Weighted
            args.append("%d" % self._d)
            args.append("%f" % self._eta)
            args.append("%f" % self._regu)
            args.append("%d" % self._max_iter)
            args.append("%d" % self._print_step)
            t1 = time()
            try:
                call(args)
            except Exception as e:
                print(str(e))
                c_flag = False
                print('./gf not found. Reverting to Python implementation. Please compile gf, place node2vec in the path and grant executable permission')
            if c_flag:
                try:
                    self._X = graph_util.loadEmbedding(embFileName)
                except FileNotFoundError:
                    self._X = np.random.randn(graph.number_of_nodes(), self._d)
                t2 = time()
                try:
                    call(["rm", embFileName])
                except:
                    pass
                return self._X, (t2 - t1)
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
        t1 = time()
        self._node_num = graph.number_of_nodes()
        self._X = 0.01 * np.random.randn(self._node_num, self._d)
        for iter_id in range(self._max_iter):
            if not iter_id % self._print_step:
                [f1, f2, f] = self._get_f_value(graph)
                print('\t\tIter id: %d, Objective: %g, f1: %g, f2: %g' % (
                    iter_id,
                    f,
                    f1,
                    f2
                ))
            for i, j, w in graph.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[i, :], self._X[j, :])) * self._X[j, :]
                term2 = self._regu * self._X[i, :]
                delPhi = term1 + term2
                self._X[i, :] -= self._eta * delPhi
        t2 = time()
        return self._X, (t2 - t1)

    def get_embedding(self):
        return self._X

    def get_edge_weight(self, i, j):
        return np.dot(self._X[i, :], self._X[j, :])

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


if __name__ == '__main__':
    # load Zachary's Karate graph
    edge_f = 'data/karate.edgelist'
    G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    graph_util.print_graph_stats(G)
    t1 = time()
    embedding = GraphFactorization(2, 100000, 1 * 10**-4, 1.0)
    embedding.learn_embedding(graph=G, edge_f=None,
                              is_weighted=True, no_python=True)
    print ('Graph Factorization:\n\tTraining time: %f' % (time() - t1))

    viz.plot_embedding2D(embedding.get_embedding(),
                         di_graph=G, node_colors=None)
    plt.show()
