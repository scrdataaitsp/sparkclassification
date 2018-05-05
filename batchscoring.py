from pyspark.ml.classification import DecisionTreeClassificationModel
import pandas as pd
import pyspark
import os
import urllib
import sys
from azureml.logging import get_azureml_logger
import createfeautures as cf
from azureml.dataprep import datasource
# initialize logger
run_logger = get_azureml_logger() 
# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('classification').getOrCreate()
# print runtime versions
modelpath = "wasbs://srramhdispark-2018-03-28t20-34-23-500z@srramstorage.blob.core.windows.net/HdiNotebooks/PySpark/bestmodel.mml"
model = DecisionTreeClassificationModel.load(modelpath)

inputdata = datasource.load_datasource('POLines.dsource')
data = inputdata.dropna(subset=['Category'])

###Get the records to be scored####
dsinput=data[data['Category']==""]
rawdata = dsinput[['Category','Scenario','Company Code','Type','PGr','Created','Short Text','Storage Location','Vendor Material Number','Base Unit of Measure','Unit of Weight','Acct Assignment Cat','Material freight grp','Plant','Profit Center']]
pdf = rawdata.toPandas()
print ('Preparing a String Column for Classification')
pdf['inputstring'] = pdf[['Scenario','Company Code','Type','PGr','Created','Short Text','Storage Location','Vendor Material Number','Base Unit of Measure','Unit of Weight','Acct Assignment Cat','Material freight grp','Plant','Profit Center']].apply(lambda x: ' , '.join(x.astype(str)), axis=1)
finaldataset = pdf[['Category','inputstring']]
df = spark.createDataFrame(finaldataset)
#Tokenize
processedData = cf.tokenizes(df)
#Score
transformedset = model.transform(processedData).select(["inputstring","prediction"])
transformedset.registerTempTable('classifieddataset')
df = spark.sql("SELECT  inputstring, \
               CASE WHEN prediction=1 THEN 'SUPPORTS'   \
               WHEN prediction=2 THEN 'COUPLERS' \
               WHEN prediction=3 THEN 'MISCELLANEOUS' \
               WHEN prediction=4 THEN 'HARDWARE' \
               WHEN prediction=5 THEN 'TIE WIRE' \
               WHEN prediction=6 THEN 'MESH' \
               WHEN prediction=7 THEN 'Not-Fab Acc' END as classified_label\
               FROM classifieddataset")

df.show(5)
filepath = "wasbs://srramhdispark-2018-03-28t20-34-23-500z@srramstorage.blob.core.windows.net/HdiNotebooks/PySpark/output.parquet"
df.write.mode("overwrite").format("parquet").save(filepath)
#Creating a hive table
df.write.mode("overwrite").saveAsTable("scoreddataset")