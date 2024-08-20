# Databricks notebook source
import mlflow

# COMMAND ----------

from mlsrc.config import get_configurations
file_path = '/dbfs/mnt/mlops/preprod_mlops/config/config.json'
model_config = get_configurations(config_file=file_path)


# COMMAND ----------

def register_model(model_exp : dict, run_id):
    register_model_name = model_exp['register_model_name']
    log_model_name = model_exp['log_model_name']
    tags = model_exp.get('tags',{})
    registered_model = mlflow.register_model(f"runs:/{run_id}/{log_model_name}", register_model_name, tags=tags)
    return  registered_model

# COMMAND ----------

run_id = '68747471bf9a49b395bd25466ce8b0b9'
model_exp = model_config['diabetes_cls_model']
registered_model = register_model(model_exp=model_exp, run_id =run_id )
registered_model
