# library doc string

'''
Predicting Customer Credit Card Churn Modules
Author: Jonathan Cabreira
Date: 11/1/2021
'''
# import libraries
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df

def perform_basic_eda(df):
     '''
    perform basic eda on df 
    input:
            df: pandas dataframe

    output:
            None
    '''
    print('DataFrame shape {}'.format(df.shape))
    print('Total of Null Features :\n{}',format(df.isnull().sum()))
    print('Statistics: \n{}.'format(df.describe()))
    
def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
     
    plotting_columns = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans", "Heatmap"]
    
    perform_basic_eda(df)
        
    for column in plotting_columns:
        plt.figure(figsize=(20,10))
        if column == 'Churn':
            df['Churn'].hist();
        elif column == 'Customer_Age':
            df['Customer_Age'].hist();
        elif column == 'Marital_Status':
            df.Marital_Status.value_counts('normalize').plot(kind='bar');
        elif column == 'Total_Trans_Ct':
            sns.distplot(df['Total_Trans_Ct']);
        elif columns == 'Heatmap':
            sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig("images/eda/{}.jpg".format(column))
        plt.close()
            

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for column in category_lst:
        column_lst = []
        column_groups = df.groupby(column).mean()['Churn']
        for val in df[column]:
            column_lst.append(column_groups.loc[val])
        df['{}_{}'.format(column, 'churn')] = category_lst
    return df
        
def keep_columns():
    '''
    Returns list of columns to keep in the dataframe
    input :
                None
                
    output : 
                List of columns to keep
    
    '''
    keep_columns =  ['Customer_Age', 'Dependent_count', 'Months_on_book',
                     'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                     'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                     'Income_Category_Churn', 'Card_Category_Churn']
    
    return  list_columns

def perform_feature_engineering(df, response = False):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = df['Churn']
    X = pd.DataFrame()
    
    keep_cols = keep_columns()
    
    X[keep_cols] = df[keep_cols]
    
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3,   random_state=42)
    
    return x_train, x_test, y_train, y_test
    
    
def random_forest_classifier(y_train, y_test):
     '''
     Perform a random forst classifier
     
    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
    output:
              y_train_preds_rf: Y training predictions of random forest model
              y_test_preds_rf: Y testing predictions of random forest model
    '''
    
    rfc = RandomForestClassifier(random_state=42)
        
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
                    }
    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    
    return y_train_preds_rf, y_test_preds_rf
    
def random_forest_report(y_train,y_test,y_train_preds_rf,y_test_preds_rf):
    '''
    Export random forest results
     
    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
              y_train_preds_rf: Y training predictions of random forest model
              y_test_preds_rf: Y testing predictions of random forest model
    output:
              None
    '''
    # scores
    plt.rc("figure", figsize=(5, 5))
    plt.text(0.01, 1.25, 'Random Forest', {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.05, str(classification_report(y_train, cy_train_preds_rf)), 
    {"fontsize": 10}, fontproperties="monospace")
    
    plt.text(0.01, 0.6, 'Random Forest Test', {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.7, str(classification_report(y_test, y_train_preds_rf)), {
    "fontsize": 10}, fontproperties="monospace")
    
    plt.axis("off")
    plt.savefig("images/results/{}.jpg".format('Random Forest') )
    plt.close() 
    
def logistic_regression_report(y_train,y_test,y_train_preds_lr,y_test_preds_lr):
    '''
    Export logistic regression results
     
    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
              y_train_preds_lr: Y training predictions of random forest model
              y_test_preds_lr: Y testing predictions of random forest model
    output:
              None
    '''
    # scores
    plt.rc("figure", figsize=(5, 5))
    plt.text(0.01, 1.25, 'Logistic Regression', {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.05, str(classification_report(y_train, cy_train_preds_lr)), 
    {"fontsize": 10}, fontproperties="monospace")
    
    plt.text(0.01, 0.6, 'Logistit Regression Test', {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.7, str(classification_report(y_test, y_train_preds_lr)), {
    "fontsize": 10}, fontproperties="monospace")
    
    plt.axis("off")
    plt.savefig("images/results/{}.jpg".format('Logistic Regression') )
    plt.close()    

    
def logistic_regression_classififer(y_train,y_test):
    '''
    Perform a logistic regression classifier
    
    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
    output:
              y_train_preds_rf: Y training predictions of random forest model
              y_test_preds_rf: Y testing predictions of random forest model
    '''
    
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    return y_train_preds_lr, y_test_preds_lr
    
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                models_lst,
                               export_classification_plots = False
                               ):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            display_classification_metrics : Bool variable [optional argument that could be used for displaying classification metrics]

    output:
             None
    '''
    #y_train_preds_rf, y_test_preds_rf = random_forest_classifier(y_train, y_test)
    #y_train_preds_lr, y_test_preds_lr = logistic_regression_classififer(y_train,y_test)
    
    random_forest_report(y_train,y_test,y_train_preds_rf,y_test_preds_rf)
    logistic_regression_report(y_train,y_test,y_train_preds_lr,y_test_preds_lr)
     
def roc_curve_plots(X_test,Y_test,models_dict):
    '''
    Plots roc curve of the models
    input:
            model_dict: dictionary with model objects containing feature_importances_
            X_test:
            Y_test:

    output:
             None
    '''
    
    for model_name, model in models.items():
        plt.figure(figsize=(20, 5))
        rc_plot = plot_roc_curve(model, X_test, y_test)
        plt.title("{} Roc Curve".format(model_name))
        plt.ylabel("True Positive Rate")
        plt.savefig("images/results/{}_Roc_Curve.jpg".format(str(model_name)) )
        plt.show()
        
     
    plt.figure(figsize=(15, 8))
    plt.title("Random Forest best Estimator Roc Curve")
    ax = plt.gca()
    disp = plot_roc_curve(models['Random Forest'].best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    rc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/{}_Roc_Curve.jpg".format(str('Random Forest best Estimator')) )
    plt.show()
    plt.close()
    
def feature_importance_plot(model, X_data, output_pth):
    
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    if not os.path(output_pth):
        os.makedirs(output_pth)
        
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth,"Feature_Importance.jpg"))
    plt.close()
        

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass