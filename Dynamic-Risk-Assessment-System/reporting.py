import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from diagnostics import model_predictions

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
conf_matrix_output_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model():
    '''
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    '''
    print('Calculating confusion matrix')

    y_pred = model_predictions(os.path.join(test_data_path, 'testdata.csv'))

    data = pd.read_csv(os.path.join(test_data_path,'testdata.csv'))
    y_true = data['exited']

    confusion_mat = confusion_matrix(y_true,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot()
    print(f'Savind confusion matrix in {conf_matrix_output_path}')
    plt.savefig(os.path.join(conf_matrix_output_path, 'confusionmatrix2.png'))


if __name__ == '__main__':
    score_model()
