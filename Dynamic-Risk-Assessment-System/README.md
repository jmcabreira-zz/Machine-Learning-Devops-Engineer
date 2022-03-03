![](images/CabreiraLogo.png)

# # Dynamic-risk-assessment-system Project:

* In this project we create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's clients
* Set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

<p align="center">
  <img  src="images/fullprocess.png">
</p>


## Project

* Data ingestion. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.
* Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.
* Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.
* Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.
Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.


## To predict:
 * The attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue.

## Tools:
* Python
