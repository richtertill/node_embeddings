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

	def __init__(self, embedding_dimension=64, distance_meassure='sigmoid', max_epoch=5000, learning_rate=1e-2, weight_decay=1e-7, display_step=250):
		''' Initialize the kl class

		Args:
			d: dimension of the embedding
			distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
		'''

		self._d = embedding_dimension
		self._distance_meassure = distance_meassure
		self._method_name = "kl"
		self._max_epoch = max_epoch
		self._learning_rate = learning_rate
		self._weight_decay = weight_decay
		self._display_step = display_step

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def set_summary_folder(self, path):
		self._summary_path = path
		self._writer = SummaryWriter(self._summary_path)

	def learn_embedding(self, AdjMat):

		self._num_nodes = AdjMat.shape[0]
		self._num_edges = AdjMat.sum()

		# Calculate degree matrix from adjacency matrix
		#npadj=np.matrix(AdjMat)
		#s=(len(npadj),len(npadj))
		#degree = np.zeros(s) #init degree matrix
		#rowsum = npadj.sum(axis=1) #get degree of each row (-> node)
		#for j in range(0, len(npadj)):
		#    degree[j,j] = rowsum[j,0]

		# Convert adjacency matrix to a CUDA Tensor
		adjmat_cuda = torch.FloatTensor(AdjMat.toarray()).cuda()
		
		#Calculate degree matrix from adjacency matrix
		degree=adjmat_cuda.sum(dim=1, dtype=torch.float)
		inv_degree=torch.diag(1/degree)
		degree=torch.diag(degree)
		       

		#### Model definition ####

		# Define the embedding matrix
		embedding_dim = 64
		emb = nn.Parameter(torch.empty(self._num_nodes, embedding_dim).normal_(0.0, 1.0))

		# Initialize the bias
		# The bias is initialized in such a way that if the dot product between two embedding vectors is 0 
		# (i.e. z_i^T z_j = 0), then their connection probability is sigmoid(b) equals to the 
		# background edge probability in the graph. This significantly speeds up training
		# TODO: WHY does it speed up the training?
		edge_proba = self._num_edges / (self._num_nodes**2 - self._num_nodes)
		bias_init = np.log(edge_proba / (1 - edge_proba))
		b = nn.Parameter(torch.Tensor([bias_init]))

		## Define different loss functions
		
		# sigmoid loss function
		#def compute_loss_sigmoid(adjmat_cuda, emb, b=0.0): 
		#    """Compute the negative log-likelihood of the KL model."""
		#    logits = emb @ emb.t() + b
		#    logits = logits.cuda()
		#    loss = F.binary_cross_entropy_with_logits(logits, adjmat_cuda, reduction='none')
			# Since we consider graphs without self-loops, we don't want to compute loss
			# for the diagonal entries of the adjacency matrix.
			# This will kill the gradients on the diagonal.
		#    loss[np.diag_indices(adjmat_cuda.shape[0])] = 0.0
		#    return loss.mean()

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
		#if self._distance_measure == 'kl':
		compute_loss = compute_loss_kl
		# if self._distance_measure == 'sigmoid':
		# 	compute_loss = compute_loss_sigmoid
		#elif self._distance_meassure == 'sigmoid':
		
		
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
				#print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')
				self._writer.add_scalar('Loss/train', loss.item(), epoch)

		# Save the embedding
		emb_np = emb.cpu().detach().numpy()
#         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)
		
		return emb_np