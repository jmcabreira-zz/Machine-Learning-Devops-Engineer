# Predict Customer Churn

- Project **Predicting Customer Churn**. 

## Project Description
In this project we move from a jupyter notebook with a data science task to production code. 

* `churn_library.py`: library containing all functions of the data science task
* `churn_script_logging_and_tests.py`: Task and logging that test the functions of the churn_library.py library and log erros that occour.
* `churn_notebook.ipynb` : The original notebook containing the initial data science project

## Running Files

All python libraries used in this repository can be `pip` installed. 

You can get the data from [Kaggle website](https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv)

You can run the the following commands in order to retrieve the results:

* test each of the functions and provide any errors to a file stored in the `logs` folder.

```
python churn_script_logging_and_tests_solution.py
```
All functions and refactored code associated with the original notebook.
```
python churn_library_solution.py
```

You can also check the pylint score, as well as perform the auto-formatting using the following commands:

```
pylint churn_library_solution.py
pylint churn_script_logging_and_tests_solution.py
```

The files here were formated using:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests_solution.py
autopep8 --in-place --aggressive --aggressive churn_library_solution.py


