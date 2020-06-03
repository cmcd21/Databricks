# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ###Data Wrangling Demo - Semi-Structured JSON Data & Log Data
# MAGIC 
# MAGIC Apache Spark is an excellent framework for wrangling, analyzing and modeling structured and unstructured data at scale

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract
# MAGIC 
# MAGIC Data on UK flight punctuality and cancellations is available on the Civil Aviation Authority [website](https://www.caa.co.uk/Data-and-analysis/UK-aviation-market/Flight-reliability/Datasets/Punctuality-data/Punctuality-statistics-2019/). 2019 data is split into 12 Monthly csv files and we have saved these to Azure Blob Storage.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Connect to data store
# MAGIC 
# MAGIC Create connection details to the storage location (blob / data lake) using Secrets which are well integrated with Azure Key Vault

# COMMAND ----------

storageAccount = "training123storage"
container = "training"
mountName = "/mnt/databricks-demo"
blobKey = dbutils.secrets.get(scope = "key-vault-secrets", key = "BlobAccessKey") #secrets integrated with Azure Key Vault
sourceString = f"wasbs://{container}@{storageAccount}.blob.core.windows.net/"
confKey = f"fs.azure.account.key.{storageAccount}.blob.core.windows.net"

spark.conf.set(confKey,blobKey) #set account credentials in notebook’s session configs

# COMMAND ----------

# MAGIC %md
# MAGIC ##### DBFS Mounts and Azure Blob Storage
# MAGIC 
# MAGIC Azure Blob Storage is the backbone of Databricks workflows and, by colocating data with Spark clusters, Databricks quickly reads from and writes to Azure Blob Storage in a distributed manner.
# MAGIC 
# MAGIC The Databricks File System (DBFS), is a layer over Azure Blob Storage that allows you to mount Blob containers, making them available to other users in your workspace and persisting the data after a cluster is shut down.

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

# MAGIC %fs ls /mnt/databricks-demo/SemiStructured/

# COMMAND ----------

# MAGIC %fs head /mnt/databricks-demo/SemiStructured/NewYorkLotteryData.json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wrangling JSON data
# MAGIC Analysing New York Lottery draws
# MAGIC 
# MAGIC  - Pyspark used to create a dataframe. A DataFrame is the most common Structured API and simply represents a table of data with rows and columns.
# MAGIC  - Lazy Evaluation means that Spark will wait until the very last moment to execute the graph of computation instructions.

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

jsonLocation = '/mnt/databricks-demo/SemiStructured/NewYorkLotteryData.json'

df = (spark.read
      .json(jsonLocation, multiLine=True)
     )

# COMMAND ----------

display(df)

# COMMAND ----------

#use the explode function to take a column that consists of arrays and create one row per value of the array
flattenedDF = df.select(explode(col("data")).alias("explodedJSON"))
display(flattenedDF)

# COMMAND ----------

#only read the columns of interest
flattenedDF.select(col("explodedJSON").getItem(8).alias("date")
                   ,col("explodedJSON").getItem(9).alias("numbers")
                  ).show(5)

# COMMAND ----------

#split the numbers into seperate columns
df = flattenedDF.select(col("explodedJSON").getItem(8).alias("date")
                        ,split(col("explodedJSON").getItem(9), " ")[0].alias("N1")
                        ,split(col("explodedJSON").getItem(9), " ")[1].alias("N2")
                        ,split(col("explodedJSON").getItem(9), " ")[2].alias("N3")
                        ,split(col("explodedJSON").getItem(9), " ")[3].alias("N4")
                        ,split(col("explodedJSON").getItem(9), " ")[4].alias("N5")
                        ,col("explodedJSON").getItem(10).alias("Bonus")
                        )
df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wrangling Log data
# MAGIC Analysing log datasets from the NASA Kennedy Space Center web sever

# COMMAND ----------

fileLocation = '/mnt/databricks-demo/SemiStructured/*.gz'

nasaDf = spark.read.text(fileLocation)
display(nasaDf)

# COMMAND ----------

#extract hostnames
host_pattern = '(^\S+\.[\S+\.]+\S+)\s'

nasaDf.select(
          col("value"),
          regexp_extract(col("value"),host_pattern, 1).alias("host")).show(5)


# COMMAND ----------

#extract datetime
ts_pattern = '\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'

nasaDf.select(
          col("value"),
          regexp_extract(col("value"),ts_pattern, 1).alias("timestamp")).show(5)

# COMMAND ----------

#extract HTTP request method, URI and Protocol
method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'

nasaDf.select(
          col("value"),
          regexp_extract(col("value"),method_uri_protocol_pattern, 2).alias("method") #change index to extract differnt part of string
          ).show(5)


# COMMAND ----------

#extracting all to create new clean dataframe with strings and interger columns
status_pattern = '\s(\d{3})\s'
content_size_pattern = '\s(\d+)$'

logDF = nasaDf.select(
                       regexp_extract(col("value"),host_pattern, 1).alias("host")
                      ,regexp_extract(col("value"),ts_pattern, 1).alias("timestamp")
                      ,regexp_extract(col("value"),method_uri_protocol_pattern, 1).alias("method")
                      ,regexp_extract(col("value"),method_uri_protocol_pattern, 2).alias("endpoint")
                      ,regexp_extract(col("value"),method_uri_protocol_pattern, 3).alias("protocol")
                      ,regexp_extract(col("value"),status_pattern, 1).cast("integer").alias("status")
                      ,regexp_extract(col("value"),content_size_pattern, 1).cast("integer").alias("content_size")
)

logDF.show(10)

# COMMAND ----------

print(logDF.count())

# COMMAND ----------

#nulls by column
logDF.select([count(when(isnull(c), c)).alias(c) for c in logDF.columns]).show(vertical=True)

# COMMAND ----------

#find bad record in status column
status_pattern = '\s(\d{3})\s'

nasaDf.filter(~nasaDf['value'].rlike(status_pattern)).show()

# COMMAND ----------

#remove bad status record
logDF = logDF.dropna("any", subset=["status"])
logDF.select([count(when(isnull(c), c)).alias(c) for c in logDF.columns]).show(vertical=True)

# COMMAND ----------

#view null records in content_size column
logDF.filter(col("content_size").isNull()).show(10)

# COMMAND ----------

#fill null records with 0, instead of dropping them
logDF = logDF.na.fill({'content_size': 0})
logDF.select([count(when(isnull(c), c)).alias(c) for c in logDF.columns]).show(vertical=True)

# COMMAND ----------

logDF = logDF.withColumn("datetime", to_timestamp('timestamp', 'dd/MMM/yyyy:HH:mm:ss'))
logDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Joining

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC 
# MAGIC By creating a temporary view, the user can query the spark dataframe using SQL.
# MAGIC 
# MAGIC Temporary views assign a name to a query that will be reused as if they were tables themselves. Unlike tables, temporary views aren't stored on disk and are visible only to the current user.
# MAGIC 
# MAGIC Databricks provides built-in easy to use visualizations for your data and you can also use other Python libraries to generate plots. 

# COMMAND ----------

#query df using SQL
df.createOrReplaceTempView("lottery")

# COMMAND ----------

# MAGIC %sql
# MAGIC --count number of rows
# MAGIC 
# MAGIC SELECT count(*)
# MAGIC FROM lottery

# COMMAND ----------

# MAGIC %sql
# MAGIC --get most common bonus number
# MAGIC 
# MAGIC SELECT bonus, count(*)
# MAGIC FROM lottery
# MAGIC GROUP BY bonus
# MAGIC ORDER BY 2 DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC --check if a duplicated draw has ever occured
# MAGIC 
# MAGIC SELECT N1, N2, N3, N4, N5, Bonus, count(*)
# MAGIC FROM lottery
# MAGIC GROUP BY N1, N2, N3, N4, N5, Bonus
# MAGIC ORDER BY 7 DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load
# MAGIC 
# MAGIC Data is written to the data warehouse (Azure Synapse Analytics) using PolyBase by:
# MAGIC - establishing a communication channel between SQL DW and Databricks 
# MAGIC - creating a temporary storage location in Azure Storage
# MAGIC - import into dw via tempDir 

# COMMAND ----------

#data warehouse connection details
tempDir = f"wasbs://{container}@{storageAccount}.blob.core.windows.net/tempDir"
jdbcHostname = "neuedakpiserver.database.windows.net"
jdbcDatabase = "neuedadw"
jdbcPort = 1433
tablename = 'demo.DatabricksFlights'
jdbcUsername = dbutils.secrets.get(scope = "key-vault-secrets", key = "SqlDwUser")
password = dbutils.secrets.get(scope = "key-vault-secrets", key = "SqlDwPassword")
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2};user=NeuedaKPIAdmin@neuedakpiserver;password={4};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;".format(jdbcHostname,jdbcPort,jdbcDatabase,jdbcUsername,password)

# COMMAND ----------

# df.write \
#   .format("com.databricks.spark.sqldw") \
#   .option("url", jdbcUrl) \
#   .option("tempDir", tempDir) \
#   .option("forwardSparkAzureStorageCredentials", "true") \
#   .option("dbTable", tablename) \
#   .mode("append") \
#   .save()