import datetime
import numpy as np
import pathlib

import gust
# import utils
from utils import graph_util

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def evaluateStaticLinkPrediction(graph, embedding_method,train_ratio=0.8,undirected=True):

	# split edges of graph into set of train and test edges
	# val_ones and val_zeros is always empty
	train_ones, val_ones, val_zeros, test_ones, test_zeros = gust.train_val_test_split_adjacency(graph, p_val=0, p_test=1-train_ratio, random_state=0, neg_mul=1,
        every_node=True, connected=False, undirected=True,use_edge_cover=True, set_ops=True, asserts=False)

	# create set of train edges which are not in the train graph nor in the test graph
	train_zeros = []
	while len(train_zeros) < len(train_ones):
		i, j = np.random.randint(0, graph.shape[0]-1, 2)
		if graph[i, j] == 0 and (i, j) not in train_zeros:
			train_zeros.append((i, j))
	train_zeros = np.array(train_zeros) 
	
	# construct a new graph which only consists of training edges
	A_train_nodes = gust.edges_to_sparse(train_ones,graph.shape[0])
	
	# learn node embeddings
	emb = embedding_method.learn_embedding(A_train_nodes)

	# TODO: save embeddings
	#np.save('ber_link_prediction_cora_embedding.npy',emb)

	# Create edge embeddings for train_ones, train_zeros, test_ones, test_zeros
	train_X = []
	train_y = []

	for nodes in train_ones:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method="average")
		train_X.append(edge_emb_one)
		train_y.append(1)

	for nodes in train_zeros:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method="average")
		train_X.append(edge_emb_one)
		train_y.append(0)

	test_X = []
	test_y = []

	for nodes in test_ones:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method="average")
		test_X.append(edge_emb_one)
		test_y.append(1)
		
	for nodes in test_zeros:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_zero = create_edge_embedding(node_emb1, node_emb2, method="average")
		test_X.append(edge_emb_zero)
		test_y.append(0)

	# Create lists to numpy arrays
	train_X = np.array(train_X)
	train_y = np.array(train_y)
	test_X = np.array(test_X)
	test_y = np.array(test_y)

	lg = LogisticRegression()
	lg.fit(train_X, train_y)
	test_preds = lg.predict_proba(test_X)
	auc_score = roc_auc_score(test_y, test_preds[:,1])

	return auc_score


def expLP(digraph, dataset_name, embedding_method, rounds,
          result_folder, train_ratio=0.8,
		  undirected=True):

	print('\tLink Prediction is started...')

	pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
	summ_file = open('%s/link_prediction_summary.lpsumm' % result_folder, 'w')
	summ_file.write(f'{dataset_name}\n')
	auc_scores = [None] * rounds

	summary_folder_extended = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
	for round_id in range(rounds):
		summary_folder_extended_round = summary_folder_extended + str(round_id+1)
		pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True) 
		embedding_method.set_summary_folder(summary_folder_extended_round)
		auc_scores[round_id] = evaluateStaticLinkPrediction(digraph, embedding_method,
                                         train_ratio=train_ratio,
                                         undirected=undirected)

	mean_auc_score = np.mean(np.array(auc_scores))
	summ_file.write(f'Mean AUC score: {str(mean_auc_score)}' )
	summ_file.write("\n")
	summ_file.close()

def create_edge_embedding(emb1, emb2, method="average"):
	if method=="average":
		return (emb1+emb2)/2