from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.feature import VectorAssembler

def tokenize(df):
    tokenizer = Tokenizer(inputCol="itemdesc", outputCol="tokenizedText")
    tokenizedData = tokenizer.transform(df)
    numFeatures = 1000
    hashingScheme = HashingTF(inputCol="tokenizedText",
                              outputCol="features",
                              numFeatures=numFeatures)
    featurizedData = hashingScheme.transform(tokenizedData)
    processedData = featurizedData.withColumn("label", featurizedData["label"]) \
                             .select(["features", "label"])
    return processedData

def tokenizes(df):
    tokenizer = Tokenizer(inputCol="inputstring", outputCol="tokenizedText")
    tokenizedData = tokenizer.transform(df)
    numFeatures = 1000
    hashingScheme = HashingTF(inputCol="tokenizedText",
                              outputCol="features",
                              numFeatures=numFeatures)
    featurizedData = hashingScheme.transform(tokenizedData)
    processedData = featurizedData.withColumn("inputstring", featurizedData["inputstring"]) \
                                  .select(["features", "inputstring"])
    return processedData