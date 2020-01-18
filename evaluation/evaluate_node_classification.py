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


def evaluateNodeClassification( emb,Y,embedding_method,round_id,train_ratio, undirected=True):
    
    writer = embedding_method.get_summary_writer()
    train_X, test_X, train_y, test_y = train_test_split(emb, Y, random_state = round_id,test_size =1-train_ratio)
    rf = RandomForestClassifier(random_state=round_id)
    rf.fit(train_X, train_y)
    test_preds = rf.predict(test_X)
    
    micro = f1_score(test_y, test_preds, average='micro')
    macro = f1_score(test_y, test_preds, average='macro')

    # write to tensorboard
    writer.add_scalar('Node Classification/F1-micro', micro, round_id)
    writer.add_scalar('Node Classification/F1-macro', macro, round_id)

    return micro, macro

def expNC(AdjMat,Y, dataset_name, embedding_method, rounds,
          result_folder, train_ratio,train_epochs,eval_epochs,
          undirected=True):

    print('\nNode classification evaluation has started...\n')

    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    with open(result_folder + '/node_classification_summary.txt', 'a') as file:
        file.write(f'{dataset_name} & {embedding_method.get_method_summary()}: \n')

        micros = [None] * rounds
        macros = [None] * rounds
        summary_folder_extended = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
        embedding_method.setup_model_input(AdjMat)
        writer = embedding_method.get_summary_writer()
        embedding_method.reset_epoch()
        for i in range(1,int(train_epochs/eval_epochs)+1):
            emb = embedding_method.learn_embedding(eval_epochs)
        for round_id in range(rounds):
            summary_folder_extended_round = summary_folder_extended + str(round_id+1)
            pathlib.Path(summary_folder_extended_round).mkdir(parents=True, exist_ok=True) 
            embedding_method.set_summary_folder(summary_folder_extended_round)
            micros[round_id], macros[round_id] = evaluateNodeClassification(
                    emb, Y, embedding_method, round_id, train_ratio)
        
        print(f'\n{rounds} rounds complete: \n')

        mean_f1_micro_score = np.mean(np.array(micros))
        mean_f1_macro_score = np.mean(np.array(macros))

        print(f'mean f1_micro score: {mean_f1_micro_score}')
        print(f'mean f1_macro score: {mean_f1_macro_score}')
        
        file.write(f'F1-micro: ')
        for micro_score in micros:
            file.write(f'{micro_score}  ')
        file.write("\n")

        file.write(f'F1-macro: ')
        for macro_score in macros:
            file.write(f'{macro_score}  ')
        file.write("\n")


        




# X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size= split_ratio)
#     for j, model in enumerate(classifiers):
#         model.fit(X_train, Y_train)
#         score = model.score(X_test, Y_test)
#         print ( h_datasets[i], h_classifiers[j])
#         print(score)