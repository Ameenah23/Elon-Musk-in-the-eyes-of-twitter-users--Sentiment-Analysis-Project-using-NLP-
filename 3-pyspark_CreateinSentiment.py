# Databricks notebook source
# MAGIC %md
# MAGIC # Sentiment Analysis: TextBlob Vs VADER Vs Flair

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook will go over the Python implementation of TextBlob, VADER, and Flair for non-model sentiment analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Install And Import Python Libraries

# COMMAND ----------

# MAGIC %md
# MAGIC For the sentiment analysis, we need to import TextBlob, SentimentIntensityAnalyzer from vaderSentiment, and TextClassifier from flair. We also need to load the English sentiment data from TextClassifier and import Sentence for text processing for the flair pre-trained model.
# MAGIC
# MAGIC To check the sentiment prediction accuracy, we need to import accuracy_score from sklearn.
# MAGIC
# MAGIC Last but not least, we set the pandas dataframe column width to be 1000, which will allow us to see more content from the review.

# COMMAND ----------

# Data processing
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn

from datetime import date
import snscrape.modules.twitter as sntwitter

from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
#import spacy
#nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

!pip install vaderSentiment

# COMMAND ----------

!pip install flair

# COMMAND ----------

# Import TextBlob
from textblob import TextBlob

# Import VADER sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import flair pre-trained sentiment model
from flair.models import TextClassifier
classifier = TextClassifier.load('en-sentiment')

# Import flair Sentence to process input text
from flair.data import Sentence

# Import accuracy_score to check performance
from sklearn.metrics import accuracy_score

# Set a wider colwith
pd.set_option('display.max_colwidth', 1000)

# COMMAND ----------

import nltk
import pandas as pd
nltk.download('vader_lexicon')

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/tweets_cleaned.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ### SparkSession

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit
from pyspark.sql.functions import pandas_udf, PandasUDFType

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .appName('WeCloud Spark Training') \
        .getOrCreate()
print('Session created')

sc = spark.sparkContext

# COMMAND ----------

df_pandas = spark.read.format('csv').options(header='true').load('dbfs:/FileStore/tables/tweets_cleaned.csv').toPandas()

# COMMAND ----------

df_pandas.head()

# COMMAND ----------

df_pandas.info()

# COMMAND ----------

# MAGIC %md
# MAGIC From the output, we can see that this data set has one columns, 328395 records, and no missing data. The tweet column is object type.

# COMMAND ----------

# MAGIC %md
# MAGIC # VADER

# COMMAND ----------

# MAGIC %md
# MAGIC VADER (Valence Aware Dictionary and sentiment Reasoner) is a Python library focusing on social media sentiments. It has a built-in algorithm to change sentiment intensity based on punctuations, slang, emojis, and acronyms.
# MAGIC
# MAGIC The output of VADER includes four scores: compound score, negative score, neural score, and positive score.
# MAGIC
# MAGIC The pos, neu, and neg represent the percentage of tokens that fall into each category, so they add up together to be 100%.
# MAGIC
# MAGIC The compound score is a single score to measure the sentiment of the text. Similar to TextBlob, it ranges from -1 (extremely negative) to 1 (extremely positive). The scores near 0 represent the neural sentiment score.
# MAGIC
# MAGIC The compound score is not a simple aggregation of the pos, neu, and neg scores. Instead, it incorporates rule-based enhancements such as punctuation amplifiers.

# COMMAND ----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

# COMMAND ----------

df_pandas.dropna(inplace=True)

# COMMAND ----------

df_pandas.iloc[0]['tweet']

# COMMAND ----------

sid.polarity_scores(df_pandas.iloc[0]['tweet'])

# COMMAND ----------

df_pandas['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# COMMAND ----------

df_pandas['scores_V'] = df_pandas['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# COMMAND ----------

df_pandas.head()

# COMMAND ----------

df_pandas['compound'] = df_pandas['scores_V'].apply(lambda d: d['compound'])

# COMMAND ----------

df_pandas['comp_vader'] = df_pandas['compound'].apply(lambda score: 'POSITIVE' if score >0 else ('NEUTRAL' if score ==0 else 'NEGATIVE'))

# COMMAND ----------

df_pandas.head()

# COMMAND ----------

df_pandas['comp_vader'].value_counts()

# COMMAND ----------

#plot a bar graph to show count of tweet sentiment
fig = plt.figure(figsize=(7,5))
color = ['green', 'grey', 'red']
df_pandas['comp_vader'].value_counts().plot(kind='bar',color = color)
plt.title('Value count of tweet polarity')
plt.ylabel('Count')
plt.xlabel('Polarity')
plt.grid(False)
plt.show()

# COMMAND ----------

#pie chart to show percentage distribution of polarity
fig = plt.figure(figsize=(7,7))
colors = ('green',  'grey', 'red')
wp={'linewidth':2, 'edgecolor': 'black'}
tags=df_pandas['comp_vader'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, 
         startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of polarity')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # TextBlob

# COMMAND ----------

# MAGIC %md
# MAGIC TextBlob is a Python library for Natural Language Processing (NLP). Sentiment analysis is one of many NLP tasks that TextBlob supports.
# MAGIC
# MAGIC The sentiment property in TextBlob returns a polarity score and a subjectivity score for the input text.
# MAGIC
# MAGIC The polarity score ranges from -1 to 1, where -1 means extremely negative, and 1 means highly positive. A score near 0 means neutral sentiment.
# MAGIC
# MAGIC The subjectivity score ranges from 0 to 1, where 0 means extremely objective and 1 means highly subjective.

# COMMAND ----------

#get subjectivity and polarity of tweets with a function
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


#get polarity with a function
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df_pandas['Subjectivity'] = df_pandas['tweet'].apply(getSubjectivity)
df_pandas['Polarity'] = df_pandas['tweet'].apply(getPolarity)

# COMMAND ----------

from textblob import TextBlob

#create a function to check negative, neutral and positive analysis
def getAnalysis(score):
    if score<0:
        return 'NEGATIVE'
    elif score ==0:
        return 'NEUTRAL'
    else:
        return 'POSITIVE'
       

# COMMAND ----------

 df_pandas['Analysis_TB'] = df_pandas['Polarity'].apply(getAnalysis)

# COMMAND ----------

df_pandas.head()

# COMMAND ----------

 df_pandas['Analysis_TB'].value_counts()

# COMMAND ----------

df_pandas['comp_vader'].value_counts()

# COMMAND ----------

#pie chart to show percentage distribution of polarity
fig = plt.figure(figsize=(7,7))
colors = ('green',  'grey', 'red')
wp={'linewidth':2, 'edgecolor': 'black'}
tags=df_pandas['Analysis_TB'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, 
         startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of polarity')

# COMMAND ----------

# MAGIC %md
# MAGIC # FLAIR

# COMMAND ----------

# MAGIC %md
# MAGIC Flair is a state-of-art NLP framework built on PyTorch. It incorporates recent researches and provides an easy way to combine different embeddings to various NLP tasks. The pre-trained sentiment model offers a tool for sentiment analysis without training a customized model.
# MAGIC
# MAGIC Unlike TextBlob and VADER that output a sentiment score between -1 and 1, flair sentiment output the predicted label with a confidence score. The confidence score ranges from 0 to 1, with 1 being very confident and 0 being very unconfident.

# COMMAND ----------

df = spark.read.format('csv').options(header='true').load('dbfs:/FileStore/tables/tweets_cleaned.csv').toPandas()
df

# COMMAND ----------

df1 = df.iloc[:10000, :]

# COMMAND ----------

df2 = df.iloc[10001:20000, :]

# COMMAND ----------



# COMMAND ----------

# Define a function to get Flair sentiment prediction score

def score_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    score = sentence.labels[0].score
    value = sentence.labels[0].value
    return score, value

# COMMAND ----------

# Get sentiment score for each review
df1['scores_flair'] = df1['tweet'].apply(lambda s: score_flair(s)[0]) 

# Predict sentiment label for each review
df1['pred_flair'] = df1['tweet'].apply(lambda s: score_flair(s)[1])

# COMMAND ----------

df1

# COMMAND ----------

# Get sentiment score for each review
df2['scores_flair'] = df2['tweet'].apply(lambda s: score_flair(s)[0]) 

# Predict sentiment label for each review
df2['pred_flair'] = df2['tweet'].apply(lambda s: score_flair(s)[1])

# COMMAND ----------

df2

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating and examining Sentiment using three techniques: TextBlob Vs VADER Vs Flair
# MAGIC ##### Notes
# MAGIC - Look at the second notebook 
# MAGIC - The jupyuter notebook file 2 contain all details and comparing between these librares. To run Flair, that took more than three hours.
# MAGIC - I chose flair to creating seniment and apply it to my data.
# MAGIC - The Vader and textblob show how to perform this step with pyspark
# MAGIC - This note book 3 is the same as file 2 but with using pyspark dataframe to go to pandas. It works well but the problem, the cluster is stopped after a half hour. Therefore I have to split the data. Every part 10,000 data took around half hour. And I have data 328,766. That meand I need around 17 hours to run all parts. No tie to do that. 

# COMMAND ----------



# COMMAND ----------


