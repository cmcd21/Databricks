# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###Example Code from the book 'Spark The Definitive Guide'

# COMMAND ----------

storageAccount = "training123storage"
container = "training"
mountName = "/mnt/databricks-demo"
blobKey = dbutils.secrets.get(scope = "key-vault-secrets", key = "BlobAccessKey") #secrets integrated with Azure Key Vault
sourceString = f"wasbs://{container}@{storageAccount}.blob.core.windows.net/"
confKey = f"fs.azure.account.key.{storageAccount}.blob.core.windows.net"

spark.conf.set(confKey,blobKey) #set account credentials in notebook’s session configs

# COMMAND ----------

try:
  dbutils.fs.unmount(mountName) # Use this to unmount as needed
except:
  print("{} already unmounted".format(mountName))

# COMMAND ----------

try:
  dbutils.fs.mount(
    source = sourceString,
    mount_point = mountName,
    extra_configs = {confKey: blobKey}
  )
except Exception as e:
  print(f"ERROR: {mountName} already mounted. Run previous cells to unmount first")

# COMMAND ----------

# MAGIC %fs ls /mnt/databricks-demo/SparkDefinitiveGuide

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ####Chapter 2 - End-to-End example

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

fileLocation = '/mnt/databricks-demo/SparkDefinitiveGuide/2015-summary.csv'

flightData2015 = (spark.read
  .option("delimiter", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv(fileLocation)
)

# COMMAND ----------

flightData2015.sort("count").explain()

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "5")

# COMMAND ----------

flightData2015.sort("count").take(2)

# COMMAND ----------

flightData2015.createOrReplaceTempView("flight_data_2015")

# COMMAND ----------

sqlWay = spark.sql("""
                   SELECT DEST_COUNTRY_NAME, count(1)
                   FROM flight_data_2015
                   GROUP BY DEST_COUNTRY_NAME
                   """)

dataFrameWay = flightData2015\
  .groupBy("DEST_COUNTRY_NAME")\
  .count()

sqlWay.explain()
dataFrameWay.explain()


# COMMAND ----------

flightData2015.select(max("count")).take(1)

# COMMAND ----------

maxSql = spark.sql("""
SELECT DEST_COUNTRY_NAME, sum(count) as destination_total
FROM flight_data_2015
GROUP BY DEST_COUNTRY_NAME
ORDER BY sum(count) DESC
LIMIT 5
""")

maxSql.show()

# COMMAND ----------

flightData2015\
  .groupBy("DEST_COUNTRY_NAME")\
  .sum("count")\
  .withColumnRenamed("sum(count)", "destination_total")\
  .sort(desc("destination_total"))\
  .limit(5)\
  .show()

# COMMAND ----------

flightData2015\
  .groupBy("DEST_COUNTRY_NAME")\
  .sum("count")\
  .withColumnRenamed("sum(count)", "destination_total")\
  .sort(desc("destination_total"))\
  .limit(5)\
  .explain()

# COMMAND ----------

