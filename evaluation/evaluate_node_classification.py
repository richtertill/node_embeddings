import datetime
import numpy as np
import pathlib

try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as lr
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split


# class TopKRanker(oneVr):
#     def predict(self, X, top_k_list):
#         assert X.shape[0] == len(top_k_list)
#         probs = np.asarray(super(TopKRanker, self).predict_proba(X))
#         prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
#             for label in labels:
#                 prediction[i, label] = 1
#         return prediction

def evaluateNodeClassification(AdjMat,Y, embedding_method, train_ratio, train_epochs, eval_epochs, undirected=True):
    
    writer = embedding_method.get_summary_writer()
    embedding_method.setup_model_input(AdjMat)

    adjmat_cpu = AdjMat
    y_cpu = Y
    train_X, test_X, train_y, test_y = train_test_split(adjmat_cpu, y_cpu, test_size=1-train_ratio)

    for i in range(1,int(train_epochs/eval_epochs)):

        emb = embedding_method.learn_embedding(eval_epochs)

        rf = RandomForestClassifier()
        rf.fit(train_X, train_y)
        test_preds = rf.predict(test_X)
        micro = f1_score(test_y, test_preds, average='micro')
        macro = f1_score(test_y, test_preds, average='macro')

        # write to tensorboard
        writer.add_scalar('Node Classification/F1-micro', micro, i*eval_epochs)
        writer.add_scalar('Node Classification/F1-macro', macro, i*eval_epochs)

    return micro, macro

def expNC(AdjMat,Y, dataset_name, embedding_method, rounds,
          result_folder, train_ratio,train_epochs,eval_epochs,
          undirected=True):

    print('\tNode classification evaluation has started...')

    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    with open(result_folder + '/node_classification_summary.txt', 'a') as file:
        file.write(f'{dataset_name} & {embedding_method.get_method_summary()}: \n')

        micros = [None] * rounds
        macros = [None] * rounds
        summary_folder_extended = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
        for round_id in range(rounds):
            summary_folder_extended_round = summary_folder_extended + str(round_id+1)
            pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True) 
            embedding_method.set_summary_folder(summary_folder_extended_round)
            embedding_method.reset_epoch()
            micros[round_id], macros[round_id] = evaluateNodeClassification(
                    AdjMat,Y, embedding_method, train_ratio, train_epochs, eval_epochs)
        

        mean_f1_micro_score = np.mean(np.array(micros))
        mean_f1_macro_score = np.mean(np.array(macros))

        file.write(f'F1-micro: {mean_f1_micro_score}\n' )
        file.write(f'F1-macro: {mean_f1_macro_score}\n' )






# X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size= split_ratio)
#     for j, model in enumerate(classifiers):
#         model.fit(X_train, Y_train)
#         score = model.score(X_test, Y_test)
#         print ( h_datasets[i], h_classifiers[j])
#         print(score)