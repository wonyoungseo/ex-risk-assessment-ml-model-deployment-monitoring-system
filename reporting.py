import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'], config['trained_model_name'])
test_data_path = os.path.join(config['test_data_path'],
                              config['test_data_name'])
feature_columns = config['feature_variable']
target_column = config['target_variable']
plot_file_name = config['plot_file_name']



##############Function for reporting
def score_model(model_path, test_data_path, feature_cols, target_col, plot_file_name):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    y_pred, y_true = model_predictions(model_path, test_data_path, feature_cols, target_col)
    cm_output = metrics.confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm_output, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm_output.shape[0]):
        for j in range(cm_output.shape[1]):
            ax.text(x=j, y=i, s=cm_output[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(config['output_model_path'], plot_file_name))





if __name__ == '__main__':
    score_model(model_path,
                test_data_path,
                feature_columns,
                target_column,
                plot_file_name)
