import datetime
import numpy as np
import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import gust
from utils import graph_util


def evaluateLinkPrediction(AdjMat,embedding_method, round_id, train_ratio, train_epochs, eval_epochs,edge_emb_method,undirected=True):
	'''

	'''
	# split edges of graph into set of train and test edges
	# val_ones and val_zeros is always empty
	train_ones, val_ones, val_zeros, test_ones, test_zeros = gust.train_val_test_split_adjacency(AdjMat, p_val=0, p_test=1-train_ratio, random_state= round_id, neg_mul=1,
        every_node=True, connected=False, undirected=True,use_edge_cover=True, set_ops=True, asserts=False)

	# create set of train edges which are not in the train graph nor in the test graph
	train_zeros = []
	while len(train_zeros) < len(train_ones):
		i, j = np.random.randint(0, AdjMat.shape[0]-1, 2)
		if AdjMat[i, j] == 0 and (i, j) not in train_zeros:
			train_zeros.append((i, j))
	train_zeros = np.array(train_zeros) 

	# construct a new graph which only consists of training edges
	A_train_nodes = gust.edges_to_sparse(train_ones,AdjMat.shape[0])

	writer = embedding_method.get_summary_writer()
	embedding_method.setup_model_input(A_train_nodes)
	emb = embedding_method.learn_embedding(train_epochs)

	# Create edge embeddings for train_ones, train_zeros, test_ones, test_zeros
	train_X = []
	train_y = []

	for nodes in train_ones:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method=edge_emb_method)
		train_X.append(edge_emb_one)
		train_y.append(1)

	for nodes in train_zeros:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method=edge_emb_method)
		train_X.append(edge_emb_one)
		train_y.append(0)

	test_X = []
	test_y = []

	for nodes in test_ones:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_one = create_edge_embedding(node_emb1, node_emb2, method=edge_emb_method)
		test_X.append(edge_emb_one)
		test_y.append(1)
			
	for nodes in test_zeros:
		node_emb1 = emb[nodes[0]]
		node_emb2 = emb[nodes[1]]
		edge_emb_zero = create_edge_embedding(node_emb1, node_emb2, method=edge_emb_method)
		test_X.append(edge_emb_zero)
		test_y.append(0)

	# Create lists to numpy arrays
	train_X = np.array(train_X)
	train_y = np.array(train_y)
	test_X = np.array(test_X)
	test_y = np.array(test_y)

	### NO LOGISTIC REGRESSION 
	lg = LogisticRegression()
	lg.fit(train_X, train_y)
	test_preds = lg.predict_proba(test_X)
	auc_score = roc_auc_score(test_y, test_preds[:,1])
	
	#test_preds = np.array([1 if edge>=0  else 0 for edge in test_X])
	#auc_score = roc_auc_score(test_y, test_preds)
	
	# write to tensorboard
	writer.add_scalar('Link prediction/AUC score', auc_score, i*eval_epochs)

# return final auc_score
	return auc_score


def expLP(AdjMat,dataset_name,embedding_method,rounds,result_folder,train_ratio,edge_emb_method,train_epochs,eval_epochs,undirected=True):

	print('\nLink prediction evaluation has started...\n')

	df = pd.read_csv(f'{result_folder}/link_prediction_results.csv')

	pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
	summary_folder_extended = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
	for round_id in range(rounds):
		summary_folder_extended_round = summary_folder_extended + str(round_id+1)
		pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True) 
		embedding_method.set_summary_folder(summary_folder_extended_round)
		embedding_method.reset_epoch()
		auc_score = evaluateLinkPrediction(AdjMat, embedding_method, round_id,
										train_ratio,train_epochs, eval_epochs, edge_emb_method,
										undirected=undirected)

		result_dict = {"embedding_method": embedding_method.get_method_summary(), "dataset": dataset_name, "run_number": round_id+1, "auc_score": auc_score}
		df = df.append(result_dict, ignore_index=True)
	df.to_csv(f'{result_folder}/link_prediction_results.csv', index=False)

def create_edge_embedding(emb1, emb2, method="average"):
	if method=="average":
		return (emb1+emb2)/2
	if method=="dot_product":
		return np.dot(emb1,emb2)