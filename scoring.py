import os
import json
import pickle

import pandas as pd
from sklearn import metrics




#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_path = os.path.join(config['output_model_path'], config['score_output_name'])
test_data_path = os.path.join(config['test_data_path'], config['test_data_name'])
model_path = os.path.join(config['output_model_path'], config['trained_model_name'])
feature_var = config['feature_variable']
target_var = config['target_variable']

def load_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def load_test_data(test_data_path, feature_col, target_col):
    df = pd.read_csv(test_data_path, encoding='utf-8', sep=',')
    return df[feature_col], df[target_col]

def cal_f1_score(y_pred, y_true):
    return metrics.f1_score(y_true, y_pred)

def write_score_output(score, output_path):
    with open(output_path, 'w') as f:
        f.write("f1_score : {}".format(score))

#################Function for model scoring
def score_model(trained_model, X_test, y_test, output_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    y_pred = trained_model.predict(X_test)
    f1_output = cal_f1_score(y_test, y_pred)
    write_score_output(f1_output, output_path)



if __name__ == "__main__":
    trained_model = load_model(model_path)
    X_test, y_test = load_test_data(test_data_path, feature_var, target_var)
    score_model(trained_model, X_test, y_test,  output_path)

