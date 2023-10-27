# Databricks notebook source
# MAGIC %md
# MAGIC # Big Data Project: Twitter Dashboard
# MAGIC #### By Ameenah Al-Haidari

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The Topic
# MAGIC # Elon Musk 
# MAGIC in the eyes of twitter users

# COMMAND ----------

# MAGIC %md
# MAGIC ![test image](files/tables/elon.png)
# MAGIC https://www.reuters.com/markets/deals/twitter-boss-elon-musk-now-comes-hard-part-2022-10-28/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project/Goals
# MAGIC
# MAGIC This is a Sentiment Analysis Project using Natural Language Processing (NLP) Techniques, PySpark. 
# MAGIC The topic is about Elon Musk. I felt it would be a good idea to measure the people’s opinion and to obtain insights into how Twitter users felt about Elon Musk. The data is collected (328,766) for only two days (Nov 21 - Nov 22), 2022.
# MAGIC
# MAGIC ### Tools
# MAGIC ##### Python Packages:
# MAGIC   - Tweepy:  to create a stream and listen to the live tweets;
# MAGIC   - Flair: to do simple sentiment analysis on tweets.
# MAGIC
# MAGIC ##### Data Processing Engine:
# MAGIC   - Apache Spark (pyspark - databricks);
# MAGIC
# MAGIC ##### Cloud Platform: AWS;
# MAGIC AWS Services:
# MAGIC   - S3: Amazon Simple Storage Service
# MAGIC   - Athena: a serverless, interactive query service to query data and analyze big data in Amazon S3 using standard SQL
# MAGIC   - QuickSight: a cloud-scale business intelligence (BI) service that you can use to deliver easy-to-understand insights.
# MAGIC
# MAGIC ### Process 
# MAGIC   - Data Processing
# MAGIC   - Creating Sentiment
# MAGIC   - Exploratory Analysis of the Data
# MAGIC   - Data Modeling
# MAGIC   - Interpreting the Data
# MAGIC
# MAGIC ### Results
# MAGIC   - Flair is the best library that can be able to guess the meaning 
# MAGIC   - Logesstic Regression is the best model that can predict the daa with Accuracy Score: 0.9395 ROC-AUC: 0.9734

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note:
# MAGIC Ameenah_ElonMusk_BigDataProject folder contains six files:
# MAGIC - The main code (databricks notebook)
# MAGIC - Creating Sentiment using pyspark (databricks notebook)
# MAGIC - Creating Sentiment using jupyter notebook
# MAGIC - DataGrip (MySQL)
# MAGIC - QuickSight DashBoard pdf
# MAGIC - QuickSight DashBoard screenshot
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Import data

# COMMAND ----------

def mount_s3_bucket(access_key, secret_key, bucket_name, mount_folder):
  ACCESS_KEY_ID = access_key
  SECRET_ACCESS_KEY = secret_key
  ENCODED_SECRET_KEY = SECRET_ACCESS_KEY.replace("/", "%2F")

  print ("Mounting", bucket_name)

  try:
    # Unmount the data in case it was already mounted.
    dbutils.fs.unmount("/mnt/%s" % mount_folder)
    
  except:
    # If it fails to unmount it most likely wasn't mounted in the first place
    print ("Directory not unmounted: ", mount_folder)
    
  finally:
    # Lastly, mount our bucket.
    dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY_ID, ENCODED_SECRET_KEY, bucket_name), "/mnt/%s" % mount_folder)
    #dbutils.fs.mount("s3a://"+ ACCESS_KEY_ID + ":" + ENCODED_SECRET_KEY + "@" + bucket_name, mount_folder)
    print ("The bucket", bucket_name, "was mounted to", mount_folder, "\n")

# COMMAND ----------

# Set AWS programmatic access credentials
# Hide this information after applied it.
ACCESS_KEY = "A.......H"
SECRET_ACCESS_KEY = "8......"

# COMMAND ----------

mount_s3_bucket(ACCESS_KEY, SECRET_ACCESS_KEY, 'weclouddata/twitter/', 'project')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Explore the folder
# MAGIC ##### Choosing Elon Musk folder

# COMMAND ----------

# MAGIC %fs ls /mnt/project/ElonMusk/

# COMMAND ----------

# MAGIC %fs ls /mnt/project/ElonMusk/2022/11/

# COMMAND ----------

# MAGIC %fs ls /mnt/project/ElonMusk/2022/11/21/

# COMMAND ----------

# MAGIC %fs ls /mnt/project/ElonMusk/2022/11/22/

# COMMAND ----------

# MAGIC %fs ls /mnt/project/ElonMusk/2022/11/22/00/

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Read the data from the topic folder

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get SparkSession and SparkContext

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC SparkContext is needed when we want to execute operations in a cluster. SparkContext tells Spark how and where to access a cluster. It is first step to connect with Apache Cluster.

# COMMAND ----------

# Read CDR data into a DF

from pyspark.sql.types import *

ElonMuskSchema = StructType([
    StructField("tweet.id", LongType(), True),
    StructField("name", StringType(), True),
    StructField("username", StringType(), True),
    StructField("tweetTEXT", StringType(), True),
    StructField("public_metrics", FloatType(), True),
    StructField("location", StringType(), True),
    StructField("author_id", IntegerType(), True),
    StructField("created_at", StringType(), True)]
)

ElonMusk = (spark.read
    .option("header", "true")
    .option("delimiter", "\t")
    .schema(ElonMuskSchema)
    .csv("/mnt/project/ElonMusk/*/*/*/*/*")
)

# COMMAND ----------

ElonMusk.cache()

# COMMAND ----------

ElonMusk.count()

# COMMAND ----------

ElonMusk.rdd.getNumPartitions()

# COMMAND ----------

ElonMusk.printSchema()

# COMMAND ----------

ElonMusk.show(5)

# COMMAND ----------

display(ElonMusk)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Raw Data to CSV File

# COMMAND ----------

ElonMusk_rawOut = "/mnt/tweetProj_ElonMusk/ElonMusk_RawData.csv"

(ElonMusk.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("header", "true")
  .mode("overwrite")               # Replace existing files
  .csv(ElonMusk_rawOut)               # Write DataFrame to csv files
)

# COMMAND ----------

# MAGIC %fs ls /mnt/ElonMusk_Project/ElonMusk_BigDataProject/ElonMusk_RawData.csv/

# COMMAND ----------

mount_s3_bucket(ACCESS_KEY, SECRET_ACCESS_KEY, 'b18-ameenah', "ElonMusk_BigDataProject")

# COMMAND ----------

# You need to save both the raw (cleaned) data and the predictions
ElonMusk.write.option('header', 'false').option('delimiter', '\t').csv("/mnt/ElonMusk_Project/ElonMusk_BigDataProject/tweetRaw_ElonMusk.csv/")  

# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import col

ElonMusk.filter(col("tweetTEXT").contains("twitter")).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Preprocessing and creating sentiment

# COMMAND ----------

ElonMusk_txt = ElonMusk.select('tweetTEXT')

# COMMAND ----------

display(ElonMusk_txt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Text Cleaning Preprocessing
# MAGIC pyspark.sql.functions.regexp_replace is used to process the text
# MAGIC - Remove URLs such as http://cnn.com
# MAGIC - Remove special characters
# MAGIC - Substituting multiple spaces with single space
# MAGIC - Lowercase all text
# MAGIC - Trim the leading/trailing whitespaces
# MAGIC

# COMMAND ----------

import pyspark.sql.functions as F

ElonMusk_txt = ElonMusk_txt.select(F.col('tweetTEXT').alias('tweet'))
display(ElonMusk_txt)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean the Data

# COMMAND ----------

tweets_clean = ElonMusk_txt.withColumn('tweet', F.regexp_replace('tweet', r"http\S+", "")) \
                    .withColumn('tweet', F.regexp_replace('tweet', r"[^a-zA-Z]", " ")) \
                    .withColumn('tweet', F.regexp_replace('tweet', r"\s+", " ")) \
                    .withColumn('tweet', F.lower('tweet')) \
                    .withColumn('tweet', F.trim('tweet')) 
display(tweets_clean)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating and examining Sentiment using three techniques: TextBlob Vs VADER Vs Flair
# MAGIC ##### Notes
# MAGIC - Look at the second and third notebooks , creating sentiment by dataframe pyspark and jupyter notebook.
# MAGIC - The jupyuter notebook file 2 contain all details and comparing between these librares. To run Flair, that took more than three hours.
# MAGIC - I chose flair to creating seniment and apply it to my data.
# MAGIC - The following textblob to show how to perform this step with pyspark
# MAGIC - The note book 3 is the same as file 2 but with using pyspark dataframe to go to pandas. It works well but the problem is stopped after a half hour. Therefore I have to split the data.   
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### TextBlob

# COMMAND ----------

from textblob import TextBlob
from pyspark.sql.functions import col, udf

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 1   #'positive'
    elif sentiment < 0:
        return -1   #'negative'
    else:
        return 0  #'neutral'

# COMMAND ----------

tw_sent = udf(lambda x: get_sentiment(x), IntegerType())

tweets_clean_sent = tweets_clean.withColumn('sentment_TB', tw_sent(col('tweet')))
tweets_clean_sent.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flair

# COMMAND ----------

tweets_sent = spark.read.option('header',True).csv('dbfs:/FileStore/tables/ElonMusk_Sentiment.csv')

# COMMAND ----------

tweets_sent.show()

# COMMAND ----------

# cache the dataframe for faster iteration
tweets_sent.cache()

# COMMAND ----------

# run the count action to materialize the cache
tweets_sent.count()

# COMMAND ----------

display(tweets_sent)

# COMMAND ----------

tweets_sent.printSchema()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Models Training used: 
# MAGIC - Naïve Bayes Model
# MAGIC - Random Forest Model
# MAGIC - Logistic Regression Classifier model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##  Putting a pipeline together
# MAGIC
# MAGIC So far, we've trained a logistic regression classifier to predict the sentiment. To make the model pipeline reusable for predicting future tweet sentiment, we can put all the transformers and estimators in a `Pipeline` object. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naïve Bayes Model

# COMMAND ----------


import pyspark
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer, NGram, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.feature import NGram, VectorAssembler, StopWordsRemover, HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------


# Use 90% cases for training, 10% cases for testing
train, test = tweets.randomSplit([0.9, 0.1], seed=20200819)

# Create transformers for the ML pipeline
tokenizer = Tokenizer(inputCol="tweet", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="1gram_idf", minDocFreq=5) 

#minDocFreq: remove sparse terms
assembler = VectorAssembler(inputCols=["1gram_idf"], outputCol="features")
label_encoder= StringIndexer(inputCol = "Sentiment", outputCol = "label")

nb = NaiveBayes(modelType="multinomial")

pipeline = Pipeline(stages=[tokenizer, stopword_remover, cv, idf, assembler, label_encoder, nb])

pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Model

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF, CountVectorizer, Tokenizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from pyspark.ml.feature import NGram, VectorAssembler, StopWordsRemover, HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.master('local[16]').getOrCreate()

print("SparkContext created")

# COMMAND ----------


# Use 90% cases for training, 10% cases for testing
train, test = tweets_sent.randomSplit([0.9, 0.1], seed=20200819)

# Create transformers for the ML pipeline
tokenizer = Tokenizer(inputCol="tweet", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="1gram_idf", minDocFreq=5) 

#minDocFreq: remove sparse terms
assembler = VectorAssembler(inputCols=["1gram_idf"], outputCol="features")
label_encoder= StringIndexer(inputCol = "Sentiment", outputCol = "label")

rf = RandomForestClassifier()

pipeline = Pipeline(stages=[tokenizer, stopword_remover, cv, idf, assembler, label_encoder, rf])

pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression Model

# COMMAND ----------

from pyspark.ml.feature import NGram, VectorAssembler, StopWordsRemover, HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use 90% cases for training, 10% cases for testing
train, test = tweets_sent.randomSplit([0.9, 0.1], seed=20200819)

# Create transformers for the ML pipeline
tokenizer = Tokenizer(inputCol="tweet", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="1gram_idf", minDocFreq=5) 

#minDocFreq: remove sparse terms
assembler = VectorAssembler(inputCols=["1gram_idf"], outputCol="features")
label_encoder= StringIndexer(inputCol = "Sentiment", outputCol = "label")

lr = LogisticRegression(maxIter=100)

pipeline = Pipeline(stages=[tokenizer, stopword_remover, cv, idf, assembler, label_encoder, lr])

pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Result
# MAGIC The best model is Logistic Regression with Accuracy Score: 0.9395
# MAGIC ROC-AUC: 0.9734

# COMMAND ----------

predictions.cache()

# COMMAND ----------

# Note: It is a number of test data
predictions.count()

# COMMAND ----------

display(predictions)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

Final_Predictions = predictions.select('tweet', 'Sentiment', 'label', 'prediction')
display(Final_Predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Note
# MAGIC drop all aray and vector columns and select the rest columns

# COMMAND ----------

Final_Predictions.printSchema()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Write Predictions Data to CSV File

# COMMAND ----------

FinalOut_Predictions = "/mnt/tweetProj_ElonMusk/EMusk_FinalPred.csv"

(Final_Predictions.write                       # Our DataFrameWriter
  .option("delimiter", "\t")  
  .option("header", "true")
  .mode("overwrite")               # Replace existing files
  .csv(FinalOut_Predictions)               # Write DataFrame to csv files
)

# COMMAND ----------

# MAGIC %fs ls /mnt/tweetProj_ElonMusk/EMusk_FinalPred.csv/

# COMMAND ----------

mount_s3_bucket(ACCESS_KEY, SECRET_ACCESS_KEY, 'b18-ameenah', "ElonMusk_BigDataProject")

# COMMAND ----------

Final_Predictions.write.option('header', 'false').option('delimiter', '\t').csv("/mnt/ElonMusk_Project/ElonMusk_BigDataProject/EMusk_FinalPred.csv/")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Ngram Features

# COMMAND ----------

from pyspark.ml.feature import NGram, VectorAssembler, StopWordsRemover, HashingTF, IDF, Tokenizer, StringIndexer, NGram, ChiSqSelector, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use 90% cases for training, 10% cases for testing
train, test = tweets.randomSplit([0.9, 0.1], seed=20200819)

# label
label_encoder= StringIndexer(inputCol = "Sentiment", outputCol = "label")

# Create transformers for the ML pipeline
tokenizer = Tokenizer(inputCol="tweet", outputCol="tokens")
stopword_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")

cv = CountVectorizer(vocabSize=2**16, inputCol="filtered", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="1gram_idf", minDocFreq=5) #minDocFreq: remove sparse terms

ngram = NGram(n=2, inputCol="filtered", outputCol="2gram")
ngram_hashingtf = HashingTF(inputCol="2gram", outputCol="2gram_tf", numFeatures=20000)
ngram_idf = IDF(inputCol='2gram_tf', outputCol="2gram_idf", minDocFreq=5) 

# Assemble all text features
assembler = VectorAssembler(inputCols=["1gram_idf", "2gram_tf"], outputCol="rawFeatures")

# Chi-square variable selection
selector = ChiSqSelector(numTopFeatures=2**14,featuresCol='rawFeatures', outputCol="features")

# Regression model estimator
lr = LogisticRegression(maxIter=100)

# Build the pipeline
pipeline = Pipeline(stages=[label_encoder, tokenizer, stopword_remover, cv, idf, ngram, ngram_hashingtf, ngram_idf, assembler, selector, lr])

# Pipeline model fitting
pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note
# MAGIC No big diffrence found when using Ngram Features. 

# COMMAND ----------

# MAGIC %md
# MAGIC The end

# COMMAND ----------


