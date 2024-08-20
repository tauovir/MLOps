# Databricks notebook source
# MAGIC %md
# MAGIC ### Load Libraries

# COMMAND ----------

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score,mean_absolute_error
import mlflow
from mlflow.models.signature import infer_signature

# COMMAND ----------

from mlsrc.config import get_configurations
file_path = '/dbfs/mnt/mlops/preprod_mlops/config/config.json'
model_config = get_configurations(config_file=file_path)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Read the data to train the Model
# MAGIC Dataset can be downloaded [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)

# COMMAND ----------

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
## Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
dataset_file = '/mnt/mlops/preprod_mlops/data/diabetes.csv'
dataset = spark.read.csv(dataset_file, header=True, inferSchema=True)
dataset_df = dataset.withColumnRenamed('Pregnancies', 'pregnant').withColumnRenamed('Glucose', 'glucose').withColumnRenamed('BloodPressure', 'bp').withColumnRenamed('SkinThickness', 'skin').withColumnRenamed('Insulin', 'insulin').withColumnRenamed('BMI', 'bmi').withColumnRenamed('DiabetesPedigreeFunction', 'pedigree').withColumnRenamed('Age', 'age').withColumnRenamed('Outcome', 'label')
pd_dataset = dataset_df.toPandas()
display(dataset_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Selection
# MAGIC Weneed to divide given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).

# COMMAND ----------

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pd_dataset[feature_cols] # Features
y = pd_dataset.label # Target variable

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting Data
# MAGIC  dividing the dataset into a training set and a test set is a good strategy.

# COMMAND ----------

def get_training_data():
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    return X_train, X_test, y_train, y_test


# COMMAND ----------

## Evaluate Metrics
import numpy as np
def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2
    


# COMMAND ----------

def moodel_training(model_exp : dict)->None:
    train_x, test_x, train_y, test_y = get_training_data()
    experient_info = mlflow.get_experiment_by_name(model_exp['exp_name'])
    if not experient_info:
        experiment_id = mlflow.create_experiment(name = model_exp['exp_name'],
                                                 #artifact_location= model_exp['artifact_location'],
                                                tags = model_exp.get('tags',{}))
    else:
        experient_info = mlflow.set_experiment(experiment_name = model_exp['exp_name'])
        experiment_id = experient_info.experiment_id

    with mlflow.start_run(experiment_id = experiment_id) as run:
        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        trained_model = clf.fit(train_x, train_y)
        #Predict the response for test dataset
        y_pred = trained_model.predict(test_x)

        # Metrics
        (rmse, mae, r2) = eval_metrics(test_y, y_pred)
        # Now Log the artifacts
        signature = infer_signature(model_input = test_x, model_output = y_pred)
        mlflow.log_param('criterion','entropy')
        mlflow.log_param('max_depth',3)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)
        mlflow.sklearn.log_model(trained_model,model_exp['log_model_name'],signature = signature)
        
        print("***************Model Logged Sucessfully ********************")
        
            

# COMMAND ----------

model_exp = model_config['diabetes_cls_model']
moodel_training(model_exp = model_exp)
