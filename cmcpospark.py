import numpy as np
import pandas as pd
import pyspark
import os
import urllib
import sys

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *

from azureml.logging import get_azureml_logger

# initialize logger
run_logger = get_azureml_logger() 

from azureml.dataprep import datasource

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('classification').getOrCreate()
# print runtime versions
print ('****************')
print ('Python version: {}'.format(sys.version))
print ('Spark version: {}'.format(spark.version))
print ('****************')
print ('***Prepare Input Data to get required attributes***')
inputdata = datasource.load_datasource('POLines.dsource')
data = inputdata.dropna(subset=['Category'])

print ('***Filtering Training + Testing + Validation records***')
dsinput=data[data['Category']!=""]
rawdata = dsinput[['Category','Scenario','Company Code','Type','PGr','Created','Short Text','Storage Location','Vendor Material Number','Base Unit of Measure','Unit of Weight','Acct Assignment Cat','Material freight grp','Plant','Profit Center']]
pdf = rawdata.toPandas()

print ('Preparing a String Column for Classification')
pdf['inputstring'] = pdf[['Scenario','Company Code','Type','PGr','Created','Short Text','Storage Location','Vendor Material Number','Base Unit of Measure','Unit of Weight','Acct Assignment Cat','Material freight grp','Plant','Profit Center']].apply(lambda x: ' , '.join(x.astype(str)), axis=1)
finaldataset = pdf[['Category','inputstring']]

print("Create Labels")
data = spark.createDataFrame(finaldataset)
data.registerTempTable("data1")
df = spark.sql("SELECT  inputstring as itemdesc, \
               CASE WHEN Category='SUPPORTS' THEN 1   \
               WHEN Category='COUPLERS' THEN 2 \
               WHEN Category='MISCELLANEOUS' THEN 3 \
               WHEN Category='HARDWARE' THEN 4 \
               WHEN Category='TIE WIRE' THEN 5 \
               WHEN Category='MESH' THEN 6 \
               WHEN Category='Not-Fab Acc' THEN 7 END as label\
               FROM data1")
print("Data to MML Spark Libraries")
df.show(5)

#Importing required mmlspark libraries
from mmlspark.TextFeaturizer import TextFeaturizer
from pyspark.ml.classification import DecisionTreeClassifier
from mmlspark.TrainClassifier import TrainClassifier
from mmlspark import ComputeModelStatistics, TrainedClassifierModel

##Featurization of text starts now
print("Generating Features")
textFeaturizer = TextFeaturizer() \
                 .setInputCol("itemdesc").setOutputCol("features") \
                 .setUseStopWordsRemover(True).setUseIDF(True).setMinDocFreq(5).setNumFeatures(1 << 10).fit(df)

processedData = textFeaturizer.transform(dataset=df)
processedData = processedData.withColumn("label", processedData["label"]) \
                             .select(["features", "label"])

print("Splitting the data into train, test sets")
train, test = processedData.randomSplit([0.70, 0.30])
#from pyspark.ml.classification import LogisticRegression

print("Fitting the model Starts")
model = TrainClassifier(model=DecisionTreeClassifier(),labelCol="label").fit(train)

print("Generating model scores with the test data")
prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
print("Best model's accuracy on validation set = "  + "{0:.2f}%".format(metrics.first()["accuracy"] * 100))

run_logger.log('Accuracy', metrics.first()["accuracy"])
#Save the trained model for scoring later
model.write().overwrite().save("wasbs://srramhdispark-2018-03-28t20-34-23-500z@srramstorage.blob.core.windows.net/HdiNotebooks/PySpark/POClassificationmmlspark.mml")
