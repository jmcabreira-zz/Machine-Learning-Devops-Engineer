# library doc string

'''
This is the module of the project that aim at finding customers who are likely to churn.
Author: Jonathan Cabreira

Date: 11/1/2021
'''
# import libraries
#import shap
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # load the dataframe and convert Attrition_Flag into numeric labels
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_basic_eda(df):
    '''
    perform basic eda on df
    input:
            df: pandas dataframe

    output:
            None
    '''
    # print basic dataframe info
    print('DataFrame shape {}'.format(df.shape))
    print('Total of Null Features :\n{}', format(df.isnull().sum()))
    print('Statistics: \n{}'.format(df.describe()))


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # columns to plot
    plotting_columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "Heatmap"]

    # perform_basic_eda(df)
    
    img_eda_pth =  "./images/eda/"
    if not os.path.exists(img_eda_pth):
        os.makedirs(img_eda_pth)

    for column in plotting_columns:
        plt.figure(figsize=(20, 10))
        # plot chrun
        if column == 'Churn':
            df['Churn'].hist()

        # plot Customer_Age
        elif column == 'Customer_Age':
            df['Customer_Age'].hist()

        # plot Marital_status
        elif column == 'Marital_Status':
            df.Marital_Status.value_counts('normalize').plot(kind='bar')

        # plot Total_Trans_Ct
        elif column == 'Total_Trans_Ct':
            sns.distplot(df['Total_Trans_Ct'])

        # plot Heatmap
        elif column == 'Heatmap':
            sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(img_eda_pth,"{}.jpg".format(column)))
        plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns
    '''
    for column in category_lst:
        column_lst = []
        column_groups = df.groupby(column).mean()[response]

        for val in df[column]:
            column_lst.append(column_groups.loc[val])

        new_col = '{}_{}'.format(column, response)
        df[new_col] = column_lst

    return df


def keep_columns():
    '''
    Returns list of columns to keep in the dataframe
    input :
                None

    output :
                List of columns to keep

    '''
    keep_columns_lst = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    return keep_columns_lst


def perform_feature_engineering(df, response):
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
    keep_cols = keep_columns()
    cat_columns = ['Gender',
                   'Education_Level',
                   'Marital_Status',
                   'Income_Category',
                   'Card_Category']

    y = df['Churn']
    X = pd.DataFrame()
    df = encoder_helper(df, cat_columns, response)

    X[keep_cols] = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def random_forest_classifier(X_train, X_test, y_train):
    '''
     Perform a random forst classifier

    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
    output:
              y_train_preds_rf: Y training predictions of random forest model
              y_test_preds_rf: Y testing predictions of random forest model
              cv_rfc: The model artifact
    '''

    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    return y_train_preds_rf, y_test_preds_rf, cv_rfc


def random_forest_report(y_train, y_test, y_train_preds_rf, y_test_preds_rf):
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

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.6, str('Random Forest Test (below) Random Forest Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    
    result_pth = "./images/results/"
    if not os.path.exists(result_pth):
        os.makedirs(result_pth)
        
    plt.savefig(os.path.join(result_pth,'rf_results.png'))
    plt.close()


def logistic_regression_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr):
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

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.6, str('Logistic Regression Test (below) Logistic Regression Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    
    result_pth = "./images/results/"
    if not os.path.exists(result_pth):
        os.makedirs(result_pth)
        
    plt.savefig(os.path.join(result_pth,'logistic_results.png'))
    plt.close()


def logistic_regression_classififer(X_train, X_test, y_train):
    '''
    Perform a logistic regression classifier

    input:
              y_train: pandas dataframe with training data
              y_test: pandas dataframe with testing data
    output:
              y_train_preds_rf: Y training predictions of random forest model
              y_test_preds_rf: Y testing predictions of random forest model
              lrc: The model artifact
    '''

    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return y_train_preds_lr, y_test_preds_lr, lrc


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
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

    random_forest_report(y_train, y_test, y_train_preds_rf, y_test_preds_rf)
    logistic_regression_report(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr)


def roc_curve_plots(X_test, y_test, models_dict):
    '''
    Plots roc curve of the models
    input:
            model_dict: dictionary with model objects containing feature_importances_
            X_test:
            y_test:

    output:
             None
    '''
    
    result_pth = './images/results/'
  
    
    for model_name, model in models_dict.items():
        plt.figure(figsize=(20, 5))
        rc_plot = plot_roc_curve(model, X_test, y_test)
        plt.title("{} Roc Curve".format(model_name))
        plt.ylabel("True Positive Rate")
        plt.savefig("images/results/{}_Roc_Curve.jpg".format(str(model_name)))
        # plt.show()

    plt.figure(figsize=(15, 8))
    plt.title("Random Forest best Estimator Roc Curve")
    ax = plt.gca()
    disp = plot_roc_curve(
        models_dict['Random Forest'].best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    rc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig( os.path.join(result_pth, "{}_Roc_Curve.jpg".format('Random Forest best Estimator')))
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
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, "Feature_Importance.jpg"))
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
    y_train_preds_rf, y_test_preds_rf, cv_rfc = random_forest_classifier(
        X_train, X_test, y_train)
    y_train_preds_lr, y_test_preds_lr, lrc = logistic_regression_classififer(
        X_train, X_test, y_train)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    models_dict = {'Random Forest': cv_rfc,
                   'Logistic Regression': lrc}

    roc_curve_plots(X_test, y_test, models_dict)
    
    result_pth = './images/results/'
    
    if not os.path.exists(result_pth):
        os.makedirs(result_pth)

    feature_importance_plot(cv_rfc, X_train, './images/results/')
    
    model_pth = "models/"
    if not os.path.exists(model_pth):
        os.makedirs(model_pth)

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    df = import_data("data/bank_data.csv")
    perform_eda(df)

    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        df, 'Churn')

    train_models(x_train_, x_test_, y_train_, y_test_)
