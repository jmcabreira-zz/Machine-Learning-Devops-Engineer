"""Data file ingestion, processing and fusion"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

#############Function for data ingestion
def merge_multiple_dataframe():
    '''check for datasets, compile them together, and write to an output file '''
    csv_files = glob.glob(input_folder_path + "/*.csv")

    # Append files
    data_list = []
    for file in csv_files:
        print(f'Ingesting file: {file}')
        data_list.append(pd.read_csv(file, index_col = None))
    df = pd.concat(data_list, axis = 0, ignore_index = True)

    # Drop duplicates
    df.drop_duplicates(inplace = True)

    # Save file
    print(f'Saving finaldata.csv and ingestedfiles.txt in : {output_folder_path} folder')
    df.to_csv(os.path.join(output_folder_path,'finaldata.csv'), index  = False)
    
    # 
    ingested_files_pth = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_pth, 'w' ) as file:
        file.write(json.dumps(csv_files))

if __name__ == '__main__':
    merge_multiple_dataframe()
