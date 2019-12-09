from abc import ABCMeta

class StaticGraphEmbedding:
	__metaclass__ = ABCMeta

	def __init__(self, d):
		'''Initialize the Embedding class

		Args:
			d: dimension of embedding
		'''
		pass

	def get_method_name(self):
		''' Returns the name for the embedding method

		Return: 
			The name of embedding
		'''		
		return ''

	def set_summary_writer(self, path):
		''' Creates a Tensorboard SummaryWriter instances
		
		Args:
			path: path to folder where experiment results are stored
		'''
		pass

	def get_summary_writer(self):
		''' Returns the Tensorboard SummaryWriter instance

		Return: 
			Instance of Tensorboard SummaryWriter which already points to the model summary folder
		'''	
		pass

	def get_method_summary(self):
		''' Returns the summary for the embedding include method name and paramater setting

		Return: 
			A summary string of the method
		'''		
		return ''

	def learn_embedding(self, graph):
		'''Learning the graph embedding from the adjcency matrix.

		Args:
			graph: the graph to embed in networkx DiGraph format
		'''
		pass

	def setup_model_input(self, adj_mat):
		pass

	def get_embedding(self):
		''' Returns the learnt embedding

		Return: 
			A numpy array of size #nodes * d
		'''
		pass
	
	def get_embedding_dim(self):
		''' Returns the size of the embedding vector

		Return: 
			A number representing the size of the embedding vector
		'''
		pass