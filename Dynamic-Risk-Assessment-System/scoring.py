from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['output_model_path'],  'trainedmodel.pkl') 
score_path = os.path.join(config['output_model_path'], 'latestscore.txt') 

#################Function for model scoring
def score_model(test_data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop(['corporation', 'exited'], axis=1)
    y_test = test_df['exited']
    
    # Read model
    with open(model_path , 'rb') as file:
        model = pickle.load(file)
    
    # model scoring
    y_pred = model.predict(X_test)
    print("pred: ", (y_pred))
    print("teste: ",(y_test.values))
    f1_score = metrics.f1_score(y_test.values, y_pred)
    print(f'F1 Score: {f1_score}')

    print(f"Savind F1 score in {score_path}")
    with open(score_path, 'w') as file:
        file.write(str(f1_score))

    return f1_score


if __name__ == '__main__':
    f1_score = score_model(test_data_path)

