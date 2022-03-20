from flask import Flask, session, jsonify, request
import json
import os

from diagnostics import dataframe_summary, missing_data_summary, outdated_packages_list, execution_time
from scoring import load_model, load_test_data, score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'], config['trained_model_name'])
test_data_path = os.path.join(config['test_data_path'],
                              config['test_data_name'])
feature_columns = config['feature_variable']
target_column = config['target_variable']



#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    model = load_model(model_path)
    dataset_path = request.get_json()['dataset_path']
    X_test, y_test = load_test_data(dataset_path, feature_columns, target_column)

    y_pred = model.predict(X_test)
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    model = load_model(model_path)
    X_test, y_test = load_test_data(test_data_path, feature_columns, target_column)
    score = score_model(model, X_test, y_test, output_path=None, write_file=False)
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    summary = dataframe_summary(test_data_path, feature_columns)
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    missing_data_result = missing_data_summary(test_data_path)
    exec_time = execution_time('training.py', 'ingestion.py')
    package_list = outdated_packages_list()
    return str(
        "execution_time: {}".format(exec_time) + '\n' +\
        "missing data: {}".format(missing_data_result) + '\n' +\
        "outdated package list: {}".format(package_list)
    )

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
