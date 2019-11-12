import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from time import time

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from utils import graph_util

class KL(StaticGraphEmbedding):

	def __init__(self, embedding_dimension=64, distance_meassure='sigmoid'):
		''' Initialize the kl class

		Args:
			d: dimension of the embedding
			distance_meassure: name of distance meassure ('sigmoid','gaussian',...)
		'''

		self._d = embedding_dimension
		self._distance_meassure = distance_meassure
		self._method_name = "kl"

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

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
			P = inv_degree.mm(adjmat_cuda)
			loss = -(P*torch.log(F.softmax(emb.mm(emb.t()),dim=1,dtype=torch.float))).mean()
			return loss
			
		# other loss function

		
		# Choose loss function
		if self._distance_measure == 'kl':
			compute_loss = compute_loss_kl
		if self._distance_measure == 'sigmoid':
			compute_loss = compute_loss_sigmoid
		#elif self._distance_meassure == 'sigmoid':
		
		
		#### Model definition end ####
		
		
		#### Learning ####
		
		# Regularize the embeddings but don't regularize the bias
		# The value of weight_decay has a significant effect on the performance of the model (don't set too high!)
		opt = torch.optim.Adam([
			{'params': [emb], 'weight_decay': 1e-7},
			{'params': [b]}],
			lr=1e-2)

		
		# Training loop
		max_epochs = 5000
		display_step = 250

		for epoch in range(max_epochs):
			opt.zero_grad()
			loss = compute_loss(adjmat_cuda, emb, b)
			loss.backward()
			opt.step()
			# Training loss is printed every display_step epochs
			if epoch % display_step == 0:
				print(f'Epoch {epoch:4d}, loss = {loss.item():.5f}')

		
		
		# Save the embedding
		emb_np = emb.cpu().detach().numpy()
#         np.savetxt('embedding_' + self._savefilesuffix + '.txt', emb_np)
		
		return emb_np

#     def get_embedding(self, filesuffix=None):
#         return self._Y if filesuffix is None else np.loadtxt(
#             'embedding_' + filesuffix + '.txt'
#         )

	# def get_edge_weight(self, i, j, embed=None, filesuffix=None):
#         if embed is None:
#             if filesuffix is None:
#                 embed = self._Y
#             else:
#                 embed = np.loadtxt('embedding_' + filesuffix + '.txt')
#         if i == j:
#             return 0
#         else:
#             S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
#             return (S_hat[i, j] + S_hat[j, i]) / 2

	# def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
#         if embed is None:
#             if filesuffix is None:
#                 embed = self._Y
#             else:
#                 embed = np.loadtxt('embedding_' + filesuffix + '.txt')
#         S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
#         return graphify(S_hat)

	# def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
#         if filesuffix is None:
#             if node_l is not None:
#                 return self._decoder.predict(
#                     embed,
#                     batch_size=self._n_batch)[:, node_l]
#             else:
#                 return self._decoder.predict(embed, batch_size=self._n_batch)
#         else:
#             try:
#                 decoder = model_from_json(
#                     open('decoder_model_' + filesuffix + '.json').read()
#                 )
#             except:
#                 print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
#                 exit()
#             try:
#                 decoder.load_weights('decoder_weights_' + filesuffix + '.hdf5')
#             except:
#                 print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
#                 exit()
#             if node_l is not None:
#                 return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
#             else:
#                 return decoder.predict(embed, batch_size=self._n_batch)


# if __name__ == '__main__':
#     # load Zachary's Karate graph
#     edge_f = 'data/karate.edgelist'
#     G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
#     G = G.to_directed()
#     res_pre = 'results/testKarate'
#     graph_util.print_graph_stats(G)
#     t1 = time()
#     embedding = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3,
#                      n_units=[50, 15], rho=0.3, n_iter=50, xeta=0.01,
#                      n_batch=500,
#                      modelfile=['./intermediate/enc_model.json',
#                                 './intermediate/dec_model.json'],
#                      weightfile=['./intermediate/enc_weights.hdf5',
#                                  './intermediate/dec_weights.hdf5'])
#     embedding.learn_embedding(graph=G, edge_f=None,
#                               is_weighted=True, no_python=True)
#     print('SDNE:\n\tTraining time: %f' % (time() - t1))

#     viz.plot_embedding2D(embedding.get_embedding(),
#                          di_graph=G, node_colors=None)
#     plt.show()