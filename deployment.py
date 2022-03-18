import os
import sys
import shutil
import json




##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'],
                          config['trained_model_name'])
score_info_path = os.path.join(config['output_model_path'],
                          config['score_output_name'])
dataset_info_path = os.path.join(config['output_folder_path'],
                                config['ingestion_record_file_name'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def deploy_model(destination_path, *args):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    for target in args:
        shutil.copy(target, destination_path)
        
        
if __name__ == "__main__":
    deploy_model(prod_deployment_path, model_path, score_info_path, dataset_info_path)

