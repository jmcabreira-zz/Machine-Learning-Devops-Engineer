from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os

from diagnostics import (
    model_predictions,
    dataframe_summary, 
    missing_data, 
    execution_time, 
    outdated_packages_list)
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    data_path = request.args.get('datapath')
    y_preds = model_predictions(data_path=data_path, )
    return str(y_preds)  #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    data_path = request.args.get('datapath')
    score = score_model(data_path)
    return str(score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    data_path = request.args.get('datapath')
    summary = dataframe_summary(data_path = data_path)
    return str(summary) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    data_path = request.args.get('datapath')
    missing = missing_data(data_path = data_path)
    runtimes = execution_time()
    outdated_packages_ = outdated_packages_list()

    output ={
         'missing values (%)': missing,
        'Runtimes': runtimes,
        'Outdated packages': outdated_packages_
        }
    return str(output) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
