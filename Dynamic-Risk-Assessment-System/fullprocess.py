

import training
import scoring
import deployment
import diagnostics
import reporting
import ast
import json
import os
import glob
import sys 
import subprocess
from scoring import score_model

##################Load config file and paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
ingestedfiles_path = os.path.join(config['prod_deployment_path'], 'ingestedfiles.txt')
ingesteddata_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
lastestscore_path = os.path.join(config['prod_deployment_path'], 'latestscore.txt')
model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 

##################Check and read new data
#first, read ingestedfiles.txt
with open(ingestedfiles_path,'r+') as f:
        ingested_files = ast.literal_eval(f.read())

# print(ingested_files)
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = glob.glob(input_folder_path + "/*.csv")
new_files = []
print(filenames)
for file in filenames:
    print(f"{file} in {input_folder_path}")
    if os.path.basename(file) not in ingested_files:
        new_files.append(file)
    else:
        pass

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(new_files) > 0:
    subprocess.run(['python3', 'ingestion.py']) 
else:
    sys.exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(lastestscore_path, 'r') as f:
    latest_score = float(f.read())

score = score_model(ingesteddata_path)

check_model_drift = score < latest_score
##################Deciding whether to proceed, part 2
#if model drift, proceed. otherwise, finish the process 
if check_model_drift == False:
    print(f'NO Model drift. Previous model F1 score was {latest_score}. New model score is {score}.')
    sys.exit()

else:
    ##################Re-deployment
    #if evidence for model drift, re-run the deployment.py script
    # Retrain and redeploy model
    print(f'Model drift has been detected.\n')
    print(f"Previous model F1 score was {latest_score}. New model score is {score}.\n")
    print("Training new model.")
    # Retrain model with latest data
    subprocess.run(['python3', 'training.py'])

    # Score model on test data
    subprocess.run(['python3', 'scoring.py'])

    # Redeploy model
    subprocess.run(['python3', 'deployment.py'])

    # Generate report
    subprocess.run(['python3', 'reporting.py'])

    # Run diagnostics
    subprocess.run(['python3', 'apicalls.py'])











