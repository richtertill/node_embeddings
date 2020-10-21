import datetime
import numpy as np
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluateNodeClassification(emb,Y,embedding_method,round_id,train_ratio, undirected=True):
    train_X, test_X, train_y, test_y = train_test_split(emb, Y, random_state = round_id,test_size =1-train_ratio,stratify=Y)
    rf = RandomForestClassifier(random_state=round_id)
    rf.fit(train_X, train_y)
    test_preds = rf.predict(test_X)
    
    acc = accuracy_score(test_y, test_preds)
    return acc

def expNC(AdjMat,Y, dataset_name, embedding_method,rounds,result_folder,train_ratio,train_epochs,eval_epochs,undirected=True):

    print('\nNode classification evaluation has started...\n')
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f'{result_folder}/node_classification_results.csv')

    # Compute embedding only once
    embedding_method.reset_epoch()
    embedding_method.setup_model_input(AdjMat)
    summary_folder = result_folder + "/train/" + str(dataset_name) +"/" + embedding_method.get_method_summary() + "/"
    embedding_method.set_summary_folder(summary_folder)
    emb = embedding_method.learn_embedding(train_epochs)
                
    for round_id in range(rounds):
        acc_score = evaluateNodeClassification(emb, Y, embedding_method, round_id, train_ratio)
        result_dict = {"embedding_method": embedding_method.get_method_summary(), "dataset": dataset_name, "run_number": round_id+1, "acc_score": acc_score}
        df = df.append(result_dict, ignore_index=True)
    df.to_csv(f'{result_folder}/node_classification_results.csv', index=False)