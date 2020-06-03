# Databricks notebook source
# MAGIC %md
# MAGIC ## Extract
# MAGIC 
# MAGIC Data on UK flight punctuality and cancellations is available on the Civil Aviation Authority [website](https://www.caa.co.uk/Data-and-analysis/UK-aviation-market/Flight-reliability/Datasets/Punctuality-data/Punctuality-statistics-2019/). 2019 data is split into 12 Monthly csv files and we have saved these to Azure Blob Storage.

# COMMAND ----------

storageAccount = "training123storage"
container = "training"
mountName = "/mnt/databricks-demo"
blobKey = dbutils.secrets.get(scope = "key-vault-secrets", key = "BlobAccessKey") #secrets integrated with Azure Key Vault
sourceString = f"wasbs://{container}@{storageAccount}.blob.core.windows.net/"
confKey = f"fs.azure.account.key.{storageAccount}.blob.core.windows.net"

spark.conf.set(confKey,blobKey) #set account credentials in notebook’s session configs

# COMMAND ----------

#Add year parameter to notebook
#dbutils.widgets.dropdown("Year", "2019", [str(year) for year in range(2011 ,2020)])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ###### DBFS Mounts and Azure Blob Storage
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

# MAGIC %fs ls /mnt/databricks-demo/flights/

# COMMAND ----------

year = getArgument("Year")
fileLocation = f"/mnt/databricks-demo/flights/sink/{year}/*.csv"

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

df = (spark.read
  .option("delimiter", ",")
  .option("header", True)
  .option("inferSchema", True)
  .csv(fileLocation)
)

# COMMAND ----------

#action - reading the data into the dataframe
display(df)

# COMMAND ----------

#ensure all months have been read
df.groupby(df.reporting_period).count().sort(asc("reporting_period")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform
# MAGIC 
# MAGIC The following PySpark code will
# MAGIC - Drop columns we don't need
# MAGIC - rename several columns
# MAGIC - create new columns to calculate actual number of flights per timeframe

# COMMAND ----------

#we can use Pandas with PySpark, Note that this is not recommended when you have to deal with fairly large dataframes, as Pandas needs to load all the data into memory.
df.describe().toPandas()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

#drop unneeded columns
columns_to_drop = ['run_date', 'previous_year_month_flights_matched', 'previous_year_month_early_to_15_mins_late_percent', 'previous_year_month_average_delay']
df = df.drop(*columns_to_drop)

# COMMAND ----------

#rename columns  
mapping = {'reporting_period': 'month', 'reporting_airport': 'airport', 'airline_name':'airline'}
df = df.select([col(c).alias(mapping.get(c, c)) for c in df.columns])
df.printSchema()

# COMMAND ----------

#get the total number of flights by aggregating the 3 columns
cols_list = ["number_flights_matched", "actual_flights_unmatched", "number_flights_cancelled"]
expression = '+'.join(cols_list)

df = df.withColumn('number_of_flights', expr(expression))

#filter out unmatched flights
df = df.filter((df.number_of_flights > 0) &(df.flights_unmatched_percent!=100))

# COMMAND ----------

df.columns[10:19]

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##### Custom Transformations with User Defined Functions
# MAGIC 
# MAGIC Spark's built-in functions provide a wide array of functionality, covering the vast majority of data transformation use cases. Often what differentiates strong Spark programmers is their ability to utilize built-in functions since Spark offers many highly optimized options to manipulate data. This matters for two reasons:<br><br>
# MAGIC 
# MAGIC - First, *built-in functions are finely tuned* so they run faster than less efficient code provided by the user.  
# MAGIC - Secondly, Spark (or, more specifically, Spark's optimization engine, the Catalyst Optimizer) knows the objective of built-in functions so it can *optimize the execution of your code by changing the order of your tasks.* 
# MAGIC 
# MAGIC In brief, use built-in functions whenever possible.

# COMMAND ----------

def PercentToNumber(col_names):
    """
    takes column that states percentage of total and creates a new column of actual
    """
    def inner(df):
        for column in col_names:
          if column.endswith('_percent'):
            newcolumn =  column[:-8]
            df = df.withColumn(newcolumn, df[column] / 100 * df['number_of_flights'])
            df = df.withColumn(newcolumn, round(df[column], 0))
            df = df.drop(column)
        return df
    return inner

# COMMAND ----------

percentColumns = df.columns[10:19]

df = PercentToNumber(percentColumns)(df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC #### Imputing Null or Missing Data
# MAGIC 
# MAGIC Null values refer to unknown or missing data as well as irrelevant responses. Strategies for dealing with this scenario include:<br><br>
# MAGIC 
# MAGIC * **Dropping these records:** Works when you do not need to use the information for downstream workloads
# MAGIC * **Adding a placeholder (e.g. `-1`):** Allows you to see missing data later on without violating a schema
# MAGIC * **Basic imputing:** Allows you to have a "best guess" of what the data could have been, often by using the mean of non-missing data
# MAGIC * **Advanced imputing:** Determines the "best guess" of what data should be using more advanced strategies such as clustering machine learning algorithms or oversampling techniques 

# COMMAND ----------

df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show(vertical=True)

# COMMAND ----------

#further investigation into nulls

#filter out cancelled flights - they can't have an average_delay_mins if they did't fly

nullsDF = df.filter((df.average_delay_mins.isNull()) & (df.flights_cancelled_percent != 100)) 

display(nullsDF)

#can see these are unmatched and cancelled flights

# COMMAND ----------

#Check for duplicate rows
distinctDF = df.distinct()
print("Distinct rows: ",distinctDF.count(),"\n"
      "Total rows: ",df.count(),"\n"
     )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyse and Visualise Data
# MAGIC 
# MAGIC Temporary views assign a name to a query that will be reused as if they were tables themselves. Unlike tables, temporary views aren't stored on disk and are visible only to the current user.
# MAGIC 
# MAGIC Databricks provides built-in easy to use visualizations for your data and you can also use other Python libraries to generate plots. 

# COMMAND ----------

#query df using SQL
df.createOrReplaceTempView("flights")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM flights

# COMMAND ----------

# MAGIC %sql
# MAGIC --most popular airlines
# MAGIC 
# MAGIC SELECT airline, count(number_of_flights)
# MAGIC FROM flights
# MAGIC GROUP BY airline
# MAGIC HAVING count(number_of_flights) > 1000

# COMMAND ----------

# MAGIC %sql
# MAGIC --airlines with most flights > 3 hrs late
# MAGIC 
# MAGIC SELECT airline, count(flights_between_181_and_360_minutes_late)
# MAGIC FROM flights
# MAGIC GROUP BY airline
# MAGIC HAVING count(number_of_flights) > 1000

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

df.write \
  .format("com.databricks.spark.sqldw") \
  .option("url", jdbcUrl) \
  .option("tempDir", tempDir) \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", tablename) \
  .mode("append") \
  .save()