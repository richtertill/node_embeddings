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
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split


def evaluateNodeClassification( emb,Y,embedding_method,round_id,train_ratio, undirected=True):
    
   
    train_X, test_X, train_y, test_y = train_test_split(emb, Y, random_state = round_id,test_size =1-train_ratio)
    rf = RandomForestClassifier(random_state=round_id)
    rf.fit(train_X, train_y)
    test_preds = rf.predict(test_X)
    
    acc = accuracy_score(test_y, test_preds)
    

    return acc



def compute_embedding(embedding_method, AdjMat, eval_epochs):
    embedding_method.reset_epoch()
    embedding_method.setup_model_input(AdjMat)
    emb = embedding_method.learn_embedding(eval_epochs)
    return emb


def set_dict(root_dict, embedding_method):
    embedding_method.set_summary_folder(root_dict)



def expNC(AdjMat,Y, dataset_name, embedding_method, rounds,
          result_folder, train_ratio,train_epochs,eval_epochs,
          undirected=True):

    print('\nNode classification evaluation has started...\n')
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

  
    with open(result_folder + '/node_classification_summary.txt', 'a') as file:
        file.write(f'{dataset_name} & {embedding_method.get_method_summary()}: \n')
        
        set_dict(result_folder, embedding_method)
        emb = compute_embedding(embedding_method, AdjMat, train_epochs)
        
        acc_score = []
        
        for round_id in range(rounds):
            acc = evaluateNodeClassification(
                    emb, Y, embedding_method, round_id, train_ratio)
            acc_score.append(acc)
    
        
        
        

        mean_acc = np.mean(np.array(acc_score))
       

        print(f'mean accuracy score: {mean_acc}')
       
        file.write(f'Accuracy: {mean_acc}\n')
        
        #return acc_score


        




# X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size= split_ratio)
#     for j, model in enumerate(classifiers):
#         model.fit(X_train, Y_train)
#         score = model.score(X_test, Y_test)
#         print ( h_datasets[i], h_classifiers[j])
#         print(score)