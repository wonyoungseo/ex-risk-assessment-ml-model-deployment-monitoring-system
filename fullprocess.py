import os
import json

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path= config['input_folder_path']
output_folder_path= config['output_folder_path']
test_data_path= config['test_data_path']
test_data_name= config['test_data_name']
output_model_path= config['output_model_path']
prod_deployment_path= config['prod_deployment_path']
ingestion_file_name= config['ingestion_file_name']
ingestion_record_file_name= config['ingestion_record_file_name']
trained_model_name= config['trained_model_name']
score_output_name = config['score_output_name']
feature_variable= config['feature_variable']
target_variable= config['target_variable']
plot_file_name = config['plot_file_name']
api_result_name = config['api_result_name']

def main():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    ingested_files =[]
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"), "r") as report_file:
        for line in report_file:
            ingested_files.append(line.split('\t')[2])

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    new_files = False
    for filename in os.listdir(input_folder_path):
        if input_folder_path + "/" + filename not in ingested_files:
            new_files = True


    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if not new_files:
        print("No new ingested data, exiting")
        return None

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    ingestion.merge_multiple_dataframe(input_folder_path,
                                       output_folder_path,
                                       ingestion_file_name,
                                       ingestion_record_file_name)

    trained_model = scoring.load_model(model_path=os.path.join(output_model_path, trained_model_name))
    X_test, y_test = scoring.load_test_data(test_data_path, feature_variable, target_variable)
    scoring.score_model()

    with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as report_file:
        old_f1 = float(report_file.read())

    with open(os.path.join(output_model_path, "latestscore.txt"), "r") as report_file:
        new_f1 = float(report_file.read())

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if new_f1 >= old_f1:
        print("Actual F1-score({}) is BETTER/EQUAL than old F1-score({}). No drift detected. Exiting".format(new_f1, old_f1))
        return None
    else:
        print("Actual F1-score({}) is WORSE than old F1-score({}). Drift detected -> Training model".format(new_f1, old_f1))
        training.train_model()

        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        deployment.deploy_model(prod_deployment_path,
                                os.path.join(config['output_model_path'], config['trained_model_name']),
                                os.path.join(config['output_model_path'], config['score_output_name']),
                                os.path.join(config['output_folder_path'], config['ingestion_record_file_name'])
        )

        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model

        diagnostics.model_predictions(os.path.join(config['output_model_path'], config['trained_model_name']),
                                      os.path.join(config['test_data_path'], config['test_data_name']),
                                      feature_variable,
                                      target_variable)
        diagnostics.dataframe_summary(os.path.join(config['test_data_path'], config['test_data_name']),
                                      feature_variable)
        diagnostics.missing_data_summary(os.path.join(config['test_data_path'], config['test_data_name']))
        diagnostics.execution_time('training.py', 'ingestion.py')
        diagnostics.outdated_packages_list()

        reporting.score_model(os.path.join(config['output_model_path'], config['trained_model_name']),
                              os.path.join(config['test_data_path'], config['test_data_name']),
                              feature_variable,
                              target_variable,
                              plot_file_name)


if __name__ == "__main__":
    main()