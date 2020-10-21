from abc import ABCMeta

class StaticGraphEmbedding:
	__metaclass__ = ABCMeta

	def __init__(self, d):
		'''Initialize the Embedding class

		Parameters
		----------
			d: dimension of embedding
		'''
		pass

	def get_method_name(self):
		''' Returns the name for the embedding method

		Return
			The name of embedding
		'''		
		return ''

	def set_summary_writer(self, path):
		''' Creates a Tensorboard SummaryWriter instances
		
		Parameters
		----------
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

	def learn_embedding(self, num_epochs):
		'''Learning the graph embedding from the adjcency matrix.

		Parameters
		----------
		num_epochs
			The number of epochs a embedding should be trained for.
		'''
		pass

	def setup_model_input(self, adj_mat):
		pass