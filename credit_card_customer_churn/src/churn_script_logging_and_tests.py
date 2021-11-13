'''
In this file we have the tests and logging associated with the churn library script

author: Jonathan Cabreira
date: 11/08/21
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
        
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        path = "./images/eda"
    except AssertionError as err:
        logging.error("Error in perform_eda function")
        
        raise err

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda function: It seems that the image "
                        "has not been saved in the eda folder.")
        raise err


def test_encoder_helper(encoder_helper,df):
    '''
    test encoder helper
    '''
    
    try:
        cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                       'Income_Category', 'Card_Category']

        df = encoder_helper(df, cat_columns, 'Churn')
    except AssertionError as err:
        logging.error("Error in encoder_helper function!")
        
        raise err

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "categorical columns transformation")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering,df):
    '''
    test perform_feature_engineering
    '''
    
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    except AssertionError as err:
        logging.error("Error in perform_feature_engineering function")
        
        raise err
        
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering function: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "Missing objects that should be returned.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        path = "./images/results/"
    except:
        logging.error("Error in train_models function!")
        
        raise err
        
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models function: Results image files not found")
        raise err

    path = "./models/"
    try:
        # Getting the list of directories
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models function: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models function: Model files not found")
        raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(cls.import_data)
    print(DATA_FRAME.shape)
    test_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)








