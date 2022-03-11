import os
import json
from typing import List
from datetime import datetime

import pandas as pd
import numpy as np



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
ingestion_record_file_name = config['ingestion_record_file_name']
ingestion_file_name = config['ingestion_file_name']



#############Function for data ingestion
def read_tabular_file(dir_name: str, file_name: str):
    return pd.read_csv(os.path.join(dir_name, file_name), sep=',', encoding='utf-8')

def clean_dataframe(df: pd.DataFrame):
    # filter out duplicates
    df = df.drop_duplicates()
    return df

def save_dataframe(df, output_dir_name: str, file_name: str):
    df.to_csv(os.path.join(output_dir_name, file_name), sep=',', encoding='utf-8', index=False)


def write_ingested_file_record(file_name: str, file_dir: str, ingested_file_loc: str, ingested_file_name: str, ingested_file_length: int):
    with open(os.path.join(file_dir, file_name), 'a') as f:
        f.write("{datetime}\t{location}\t{filename}\t{length}\n".format(
            datetime=datetime.now(),
            location=ingested_file_loc,
            filename=ingested_file_name,
            length=ingested_file_length
        ))


def merge_multiple_dataframe(input_folder_dir: str,
                             output_folder_dir: str,
                             output_file_name: str,
                             record_file_name: str,
                             data_cols: List[str] = ["corporation", "lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"]):
    # check for datasets, compile them together, and write to an output file

    file_name_ls = os.listdir(input_folder_dir)

    df_list = pd.DataFrame(columns=data_cols)
    for file_name in file_name_ls:
        df = read_tabular_file(input_folder_dir, file_name)
        df_list = df_list.append(df)
        write_ingested_file_record(record_file_name, output_folder_dir,
                                   input_folder_dir, file_name, len(df))

    df_list = clean_dataframe(df_list)
    save_dataframe(df_list, output_folder_path, output_file_name)


if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path, output_folder_path, ingestion_file_name, ingestion_record_file_name)
