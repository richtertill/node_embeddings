import sys
sys.path.insert(0, '../')

# import embedding methods
from embedding.bernoulli import Bernoulli
from embedding.KL import KL

# import evaluation methods
from evaluation import evaluate_link_prediction

# import utils
from utils import graph_util
from utils import plot_util


# Experiment 1

exp = {
    "emb_dim": 64,
    "max_epoch": 500,
    "learning_rate": 1e-2, #Adam
    "weight_decay": 1e-7,
    "link_prediction": True,
    "number_rounds_link_prediction": 3,
    "link_prediction_train_ratio": 0.8,
    "node_classification": False,
    "number_rounds_node_classification": 3,
    "node_classification_train_ratio": 0.8
}


# pick datasets
datasets = ["conra", "hvr"]

# initialize embedding methods
b = Bernoulli(embedding_dimension=exp["emb_dim"], distance_meassure='sigmoid', max_epoch=exp["max_epoch"])
kl = KL(embedding_dimension=exp["emb_dim"], distance_meassure='sigmoid',max_epoch=exp["max_epoch"])

embedding_methods = [b]

# setup folders to store experiment setup summary and results
result_folder = plot_util.setup_folders_and_summary_files(exp, datasets, embedding_methods)
print(f'The results of the current experiment are stored at experiments/{result_folder}')


for dataset in datasets:
    
    # load dataset
    A, y = graph_util.load_dataset(dataset)
    
    for embedding_method in embedding_methods:
        
        # do link prediction
		if(exp["link_prediction"]):
        	link_prediction_folder = result_folder + "/link_prediction"
        	evaluate_link_prediction.expLP(A,dataset,embedding_method,exp["number_rounds_link_prediction"],link_prediction_folder, train_ratio=exp["link_prediction_train_ratio"],undirected=True)
        
        # do node classification
		if(exp["node_classification"]):
			node_classification_folder = result_folder + "/node_classification"
        	evaluate_node_classification.expNC(A,dataset,embedding_method,exp["number_rounds_node_classification"],node_classification_folder, train_ratio=exp["node_classification_train_ratio"],undirected=True)