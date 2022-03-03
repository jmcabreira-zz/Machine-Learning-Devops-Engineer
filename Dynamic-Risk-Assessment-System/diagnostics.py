
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path =  os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 


##################Function to get model predictions
def model_predictions(data_path):
    #read the deployed model and a test dataset, calculate predictions
    data = pd.read_csv(data_path)
    y = data['exited']
    X = data.drop(['corporation', 'exited'], axis=1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    return y_pred.tolist()

##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    df = pd.read_csv(data_path)
    df = df.drop(['corporation', 'exited'], axis=1)

    stats_dicst = {}
    stats_dicst['col_means'] = dict(df.mean())
    stats_dicst['col_medians'] = dict(df.median())
    stats_dicst['col_std'] = dict(df.std())
    return stats_dicst

##################Function to get missing data
def missing_data(data_path):
    df = pd.read_csv(data_path)
    missing_percent = list((df.isna().sum())/len(df.index))
    return missing_percent

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime

    return round(timing, 2)

##################Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.run(['pip', 'list', '--outdated', '--format', 'json'], capture_output=True).stdout
    outdated = outdated.decode('utf8').replace("'", '"')
    outdated_list = json.loads(outdated)
    return outdated_list

if __name__ == '__main__':
    model_predictions(os.path.join(test_data_path, 'testdata.csv'))
    dataframe_summary(os.path.join(dataset_csv_path, 'finaldata.csv'))
    missing_data(os.path.join(dataset_csv_path, 'finaldata.csv'))
    execution_time()
    outdated_packages_list()





    
