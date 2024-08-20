# Databricks notebook source
# MAGIC %md
# MAGIC ### Setup location and Dataset
# MAGIC - Setup mount location
# MAGIC - Data Location
# MAGIC - Log file location
# MAGIC - Database/Schema name

# COMMAND ----------

dbutils.widgets.dropdown("Environment", "preprod", ["preprod", "prod"])
dbutils.widgets.text("mount_location","")
dbutils.widgets.text("db_name","")


# COMMAND ----------

# MOUNT_LOCATION = '/mnt/data/dbdemos/dbdemos-dbfs'
project_dir = dbutils.widgets.get('mount_location') + 'preprod_mlops'
log_dir = project_dir + '/log'
config_dir = project_dir + '/config'
db_dir = project_dir + '/db'
data_dir = project_dir + '/data'
model_artifact = project_dir + '/artifact'

# COMMAND ----------

display(dbutils.fs.ls(project_dir))

# COMMAND ----------

# Create Diretorys
def create_dir(dir_names : list[str]) -> None:
    for dir_name in dir_names:
        dbutils.fs.mkdirs(dir_name)


# COMMAND ----------

create_dir([project_dir])

# COMMAND ----------

dir_names = [log_dir, config_dir, data_dir, db_dir,model_artifact]
create_dir(dir_names)

# COMMAND ----------

# Copy Training Data to data_dir
dbutils.fs.cp("file:/Workspace/Users/<email>/MLFlow/data/diabetes.csv", data_dir)

# COMMAND ----------

## Copy Configuration file
dbutils.fs.cp("file:/Workspace/Users/<email>/MLFlow/config/config.json", config_dir)

# COMMAND ----------

class DBSetup:
    def __init__(self, db_name):
        self.db_name = db_name
    def create_db(self,schema_location : None):
        if schema_location:
            sql_query = f"""
            CREATE SCHEMA IF NOT EXISTS {self.db_name} COMMENT 'DATABASE Schema' LOCATION '{schema_location}'
            """
        else:
            sql_query = f"""
            CREATE SCHEMA IF NOT EXISTS {self.db_name} COMMENT 'DATABASE Schema'
            """
        spark.sql(sql_query)
        print(f"Database {self.db_name} created")
    def drop_db(self):
        spark.sql(f"DROP DATABASE IF EXISTS {self.db_name} CASCADE")
        print(f"Database {self.db_name} dropped")
    def create_table(self, table_name):
        pass

# COMMAND ----------

## Create Database
#db_name = dbutils.widgets.getArgument("db_name")
db_name = dbutils.widgets.get("db_name")
obj = DBSetup(db_name)
obj.create_db(schema_location = db_dir)


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE SCHEMA EXTENDED mlops_db_preprod
