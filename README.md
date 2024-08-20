# MLOps
Machine learning operation with MLFlow
This git repo represent ML operation with Mlflow in Databrciks
## Description
The folder structure are describe below
- Config : This folder contain configuration related information
- data : It contains diabetes dataset
- mlops :  This folder contain project work
## Steps to execute mloperations
- 1: mounting_adls : Run this notebook to mount ADLSGen2 to Databrciks filesystem
- 2: Setup :  Run this notebook to setup environment
- 3: model_training : Run this notebook to train model and MLFlow Experement
- 4 : model_registration :  Once exeperiment finalized,copy run_id from experiment and run model_registration notebook to register model in model registry
- 5 : model_batch_inference.py : This script load model from model registry and predict on sample feature and store inference in delta table.
Note : Before running model_batch_inference.py, place the register model in Staging stage (check model interface.)

