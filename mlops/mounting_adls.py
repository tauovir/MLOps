# Databricks notebook source
dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Mount the Azure ADL SGen2 to Databrciks file system

# COMMAND ----------

storageAccountName = "mlstorage101" # Storage account name
## Azure Azcess Key
storageAccountAccessKey = 'storageAccountAccessKey'
blobContainerName = "modelcnt" # container name
mountPoint = "/mnt/mlops/"
if not any(mount.mountPoint == mountPoint for mount in dbutils.fs.mounts()):
  try:
    dbutils.fs.mount(
      source = "wasbs://{}@{}.blob.core.windows.net".format(blobContainerName, storageAccountName),
      mount_point = mountPoint,
      extra_configs = {'fs.azure.account.key.' + storageAccountName + '.blob.core.windows.net': storageAccountAccessKey}
   
    )
    print("mount succeeded!")
  except Exception as e:
    print("mount exception", e)

# COMMAND ----------

#dbutils.fs.mounts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unmount a mount point

# COMMAND ----------

dbutils.fs.unmount("/mnt/mlops/")

# COMMAND ----------


