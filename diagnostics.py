import os
import sys
import subprocess
import json
import timeit
import pickle

import pandas as pd
import numpy as np


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'], config['trained_model_name'])
test_data_path = os.path.join(config['test_data_path'],
                              config['test_data_name'])
feature_columns = config['feature_variable']
target_column = config['target_variable']

def load_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def load_test_data(test_data_path, feature_col, target_col):
    df = pd.read_csv(test_data_path, encoding='utf-8', sep=',')
    return df[feature_col], df[target_col]

##################Function to get model predictions
def model_predictions(model_path, dataset_path, feature_col, target_col):
    # read the deployed model and a test dataset, calculate predictions
    model = load_model(model_path)
    X_test, y_test = load_test_data(dataset_path, feature_col, target_col)

    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test), "length of model prediction output does not match the length of input dataset"

    return y_pred

##################Function to get summary statistics
def missing_data_summary(test_data_path):
    # count missing data count
    df = pd.read_csv(test_data_path, encoding='utf-8', sep=',')

    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na

        result.append([column, str(int(count_na / count_total * 100)) + "%"])

    return str(result)


def dataframe_summary(test_data_path, feature_cols):
    #calculate summary statistics here
    df = pd.read_csv(test_data_path, encoding='utf-8', sep=',')

    result = []
    for column in feature_cols:
        result.append([column, "mean", df[column].mean()])
        result.append([column, "median", df[column].median()])
        result.append([column, "standard deviation", df[column].std()])

    return result



##################Function to get timings
def execution_time(*args):
    #calculate timing of training.py and ingestion.py

    result = []
    for procedure in args:
        start_time = timeit.default_timer()
        os.system('python3 {}'.format(procedure))
        timing = timeit.default_timer() - start_time
        result.append([procedure, timing])
    return str(result)

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions(model_path, test_data_path, feature_columns, target_column)
    dataframe_summary(test_data_path, feature_columns)
    execution_time('training.py', 'ingestion.py')
    outdated_packages_list()





    
