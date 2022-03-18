import os
import json
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], config['ingestion_file_name'])
model_path = os.path.join(config['output_model_path'], config['trained_model_name'])

def load_data(data_path: str):
    return pd.read_csv(data_path, encoding='utf-8', sep=',')

#################Function for training the model
def train_model(train_X, train_y, model_path):
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0,
                               class_weight=None,
                               dual=False,
                               fit_intercept=True,
                               intercept_scaling=1,
                               l1_ratio=None,
                               max_iter=100,
                               multi_class='auto',
                               n_jobs=None,
                               penalty='l2',
                               random_state=0,
                               solver='liblinear',
                               tol=0.0001,
                               verbose=0,
                               warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(train_X, train_y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(os.path.join(model_path), 'wb'))

if __name__ == "__main__":

    df = load_data(dataset_csv_path)
    train_model(df[config['feature_variable']], df[config['target_variable']], model_path=model_path)