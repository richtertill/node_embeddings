import datetime
import numpy as np
import pathlib

try: import cPickle as pickle
except: import pickle

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import numpy as np


def evaluateNodeClassification(labels_true, emb, train_ratio, round_id, undirected=True):

    train_X, test_X, train_y, test_y = train_test_split(emb, labels_true, test_size=1-train_ratio)

    rf = RandomForestClassifier(random_state=round_id)
    rf.fit(train_X, train_y)

    test_preds = rf.predict(test_X)

    micro = f1_score(test_y, test_preds, average='micro')
    macro = f1_score(test_y, test_preds, average='macro')
    acc = accuracy_score(test_y, test_preds)

    return micro, macro, acc


def compute_embedding(embedding_method,AdjMat,eval_epochs ):
    embedding_method.reset_epoch()
    embedding_method.setup_model_input(AdjMat)
    emb = embedding_method.learn_embedding(eval_epochs)
    return emb


def set_dict(root_dict,round_id, embedding_method):
    summary_folder_extended_round = root_dict + str(round_id + 1)
    pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True)
    embedding_method.set_summary_folder(summary_folder_extended_round)


def exp_Node_Classification(AdjMat, Y, dataset_name, embedding_method, rounds,
          result_folder, eval_epochs,
          undirected=True):
    print('\tNode classification evaluation has started...')

    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

    with open(result_folder + '/node_classification_summary.txt', 'a') as file:

        file.write(f'{dataset_name} & {embedding_method.get_method_summary()}: \n')
        summary_folder_extended = result_folder + "/train/" + str(
            dataset_name) + "/" + embedding_method.get_method_summary() + "/"

        accuracy = []
        micros = []
        macros = []

        emb = compute_embedding(embedding_method, AdjMat, eval_epochs)

        for round_id in range(rounds):

            set_dict(summary_folder_extended,round_id,embedding_method)

            micro,macro,acc = evaluateNodeClassificationg(
               Y, emb, round_id)

            accuracy.append(acc)
            micros.append(micro)
            macros.append(macro)

            writer = embedding_method.get_summary_writer()
            writer.add_scalar('Node Classification/F1-micro', micro, round_id)
            writer.add_scalar('Node Classification/F1-macro', macro, round_id)
            writer.add_scalar('Node Classification/Accuracy', acc, round_id)

        mean_f1_micro_score = np.mean(np.array(micros))
        mean_f1_macro_score = np.mean(np.array(macros))
        mean_accuracy= np.mean(np.array(accuracy))

        file.write(f'F1-micro: {mean_f1_micro_score}\n')
        file.write(f'F1-macro: {mean_f1_macro_score}\n')
        file.write(f'Accuracy: {mean_accuracy}\n')

        #plot_boxplot(norm_MI_score, plot_boxplot=True)