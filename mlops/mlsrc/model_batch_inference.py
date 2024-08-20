import mlflow
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime as dt
from pyspark.sql.functions import lit
from custom_log import CustomLogger
from config import get_configurations

log_obj = CustomLogger()
logger = log_obj.get_logger(logfile_prefix = 'logging', logger_name = 'custom_log',enable_file_log = True)

def get_feature_data() -> pd.DataFrame:
    # load dataset
    dataset_file = '/mnt/mlops/preprod_mlops/data/sample_inference.csv'
    dataset = spark.read. csv(dataset_file, header=True)
    pd_dataset = dataset. toPandas()
    feature_cols = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']
    feature_dataset = pd_dataset [feature_cols] # Features
    return feature_dataset, pd_dataset

def get_register_model (model_name: str, version : str = '1'):

    """
    Description: This function return mlflow registered model from registry that are registered.
    params:
    model_name : Register model name
    version : default version 1, stages can also be used example :Production', 'Staging', 'Archived'
    """
    model_uri = f"models:/{model_name}/{version}"
    logger.info(f"model_uri: {model_uri}")
    model = mlflow.sklearn. load_model(model_uri)

    return model

def get_register_model_by_runid (run_id : str, register_model_name : str):

    logged_model = f'runs:/{run_id} /{ register_model_name}'
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.Load_model(logged_model)
    return model 


def main_job():
    """
    Descript ion: This function work as job and executed from the workflow job.
    This function get configuration from json file, get registered model, fetch feature data and predict,
    the predicted values along with additional metadata will be stored in delta table.
    Job param:
    config.j son file with full path.
        """
    logger. info ('Job started')
    if len(sys.argv) < 2:
        logger.error(f'Job parameters required:{sys.argv}')
        raise ValueError("J ob parameters required")
    
    config_file_path = sys. argv [1]
    config_file_path = '/dbfs/mnt/mlops/preprod_mlops/config/config.json'
    logger.info(f'config_file:{config_file_path}')
    configurations = get_configurations(config_file = config_file_path)
    #logger. info (f ' Configuration Locaded : {configurations} ')
    price_model_conf = configurations['diabetes_cls_model']
    log_file_path = f"{configurations['mount_location']}log"
    # Get Feature data
    feature_dataset, batch_dataset = get_feature_data()
    if feature_dataset.empty:
        logger.info('feature dataset found empty, process terminated.')
        log_obj.save_logs( log_file_path = log_file_path)
        return {"status" : 1, "message" : "feature dataset found empty, process te rminated."}
    logger.info('Received feature dataset') 
    ## Register Model stage/version

    model_version = price_model_conf.get ('register_model_stage', '1')
    trained_model = get_register_model(model_name = price_model_conf['register_model_name'],version = model_version)
    # Prediction for output
    logger.info('Registered model loaded successfully')
    prediction_result = trained_model.predict(feature_dataset)
    logger.info(' Price Model predicted successfully')

    batch_dataset['prediction'] = prediction_result
    batch_inference = spark. createDataFrame (batch_dataset)
    logger.info(f'Batch inference record count:{batch_inference.count()}')
    # Current date and metadata
    cur_datetime = dt.today().strftime('%Y-%m-%d %H:%M:%S')
    meta_data = {"pred_register_model": price_model_conf['register_model_name'],"pred_model_version": model_version,
                 "rec_src":"kaggle"}
    meta_data = json.dumps(meta_data)
    batch_inference = batch_inference.withColumn('meta_data',lit(meta_data)).withColumn('created_date',lit(cur_datetime))
    table_name = f"{configurations ['schema_name']}.{price_model_conf['inference_table']}"
    batch_inference.write.mode("append").saveAsTable(table_name)
    logger.info(f"Batch inference saved in Table : {table_name}")
    # # log_path will come from json_config
    print ("========log_file_path:", log_file_path)

    log_obj.save_logs (log_file_path = log_file_path)
    print ("Job finished sucessfully'")

    return {"status": 0, "message":"Job finished successfully"}

if __name__ == "__main__":
    dbutils.library.restartPython()
    main_job()
