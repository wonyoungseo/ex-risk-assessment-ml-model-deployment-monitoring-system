from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
dataset_csv_name = config['ingestion_file_name']
trained_model_name = config['trained_model_name']

def load_data(path: str, filename: str):
    file_dir = os.path.join(path, filename)
    return pd.read_csv(file_dir)

#################Function for training the model
def train_model(train_X, train_y, model_path, model_name):
    
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
    pickle.dump(model, open(os.path.join(model_path, model_name), 'wb'))

if __name__ == "__main__":

    df = load_data(dataset_csv_path, dataset_csv_name)
    train_model(df.drop(columns=['corporation', 'exited']), df['exited'],
                model_path=model_path, model_name=trained_model_name)