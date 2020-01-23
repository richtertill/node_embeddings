import datetime
import numpy as np
import pathlib

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

import seaborn as sns


def evaluateNodeClustering(labels_true, emb, round_id, undirected=True):
    n_cluster = len(set(labels_true))
    model = KMeans(n_clusters=n_cluster, random_state=round_id, init='k-means++').fit(emb)
    labels = model.labels_
    norm_mutual_info = normalized_mutual_info_score(labels_true, labels)
    print (normalized_mutual_info_score(labels_true, labels))
    return norm_mutual_info


def compute_embedding(embedding_method, AdjMat, eval_epochs):
    embedding_method.reset_epoch()
    embedding_method.setup_model_input(AdjMat)
    emb = embedding_method.learn_embedding(eval_epochs)
    return emb


def set_dict(root_dict, embedding_method):
    #summary_folder_extended_round = root_dict + str(1)
    #pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True)
    embedding_method.set_summary_folder(root_dict)


def plot_boxplot(data, plot_boxplot=True):
    sns.set_style('whitegrid')
    sns.boxplot(data=data)


def exp_Node_Clustering(AdjMat, Y, dataset_name, embedding_method, rounds,
                        result_folder, train_epochs, eval_epochs,
                        undirected=True):
    print('\nNode clustering evaluation has started...\n')
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    with open(result_folder + '/node_clustering_summary.txt', 'a') as file:
        file.write(f'{dataset_name} & {embedding_method.get_method_summary()}: \n')
        set_dict(result_folder, embedding_method)
        emb = compute_embedding(embedding_method, AdjMat, train_epochs)
        norm_MI_score = []
        for round_id in range(rounds):
            norm_mutual_info = evaluateNodeClustering(
                Y, emb, round_id)
            norm_MI_score.append(norm_mutual_info)
     
        mean_norm_MI_score = np.mean(np.array(norm_MI_score))
        file.write(f'Normalized_mutual_information: {mean_norm_MI_score}\n')
        # plot_boxplot(norm_MI_score, plot_boxplot=True)
        return norm_MI_score