import createfeautures as cf
import numpy as np
import pandas as pd
import pyspark
import os
import urllib
import sys
from pyspark.sql.functions import *
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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

processedData = cf.tokenize(df)

##Build a Decision Tree Model

train, test, validation = processedData.randomSplit([0.60, 0.20, 0.20], seed=123)
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)
dtModel = dt.fit(train)
print ("numNodes = ", dtModel.numNodes)
print ("depth = ", dtModel.depth)
# Evaluate model
evaluator = MulticlassClassificationEvaluator()
predictions = dtModel.transform(test)
evaluator.evaluate(predictions)

# Select the best model tweaking the depth
rHyperParams = [3,4,5,6]
dtrees = [DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=hyperParam)         
                       for hyperParam in rHyperParams]

metrics = []
models = []

for learner in dtrees:
    model = learner.fit(train)
    models.append(model)
    scored_data = model.transform(test)
    print(evaluator.evaluate(scored_data))
    metrics.append(evaluator.evaluate(scored_data))

print("####Printing metrics list")
print(metrics)
print(type(metrics))
import builtins as b ##Max does not work in pyspark there is a bug
best_metric = b.max(metrics)
best_model = models[metrics.index(best_metric)]

print("Best Metric achieved ::: " + "{0:.2f}%".format(best_metric * 100))

# Save the best model
best_model.write().overwrite().save("wasbs://srramhdispark-2018-03-28t20-34-23-500z@srramstorage.blob.core.windows.net/HdiNotebooks/PySpark/bestmodel.mml")
# Get AUC on the validation dataset
scored_val = best_model.transform(validation)
print("Best Model Evaluation Results")
print(evaluator.evaluate(scored_val))
run_logger.log('Evaluated Metric', evaluator.evaluate(scored_val))