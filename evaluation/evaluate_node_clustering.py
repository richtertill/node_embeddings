import datetime
import numpy as np
import pandas as pd
import pathlib
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def evaluateNodeClustering(labels_true, emb, round_id, undirected=True):
    n_cluster = len(set(labels_true))
    model = KMeans(n_clusters=n_cluster, random_state=round_id, init='k-means++').fit(emb)
    labels = model.labels_
    norm_mutual_info = normalized_mutual_info_score(labels_true, labels)
    # print (normalized_mutual_info_score(labels_true, labels))
    return norm_mutual_info

def exp_Node_Clustering(AdjMat, Y, dataset_name, embedding_method, rounds,
                        result_folder, train_epochs, eval_epochs,
                        undirected=True):
    print('\nNode clustering evaluation has started...\n')
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f'{result_folder}/node_clustering_results.csv')

    embedding_method.reset_epoch()
    embedding_method.setup_model_input(AdjMat)
    summary_folder = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
    embedding_method.set_summary_folder(summary_folder)
    emb = embedding_method.learn_embedding(train_epochs)

    for round_id in range(rounds):
        nmi_score = evaluateNodeClustering(Y, emb, round_id)
        result_dict = {"embedding_method": embedding_method.get_method_summary(), "dataset": dataset_name, "run_number": round_id+1, "nmi_score": nmi_score}
        df = df.append(result_dict, ignore_index=True)
    df.to_csv(f'{result_folder}/node_clustering_results.csv', index=False)