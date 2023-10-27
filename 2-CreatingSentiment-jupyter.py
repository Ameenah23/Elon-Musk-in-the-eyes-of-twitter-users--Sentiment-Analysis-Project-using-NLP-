# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sentiment Analysis: TextBlob Vs VADER Vs Flair
# MAGIC
# MAGIC By Ameenah Al-Haidari

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

from datetime import date
#import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

# Data processing
import pandas as pd
import numpy as np
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

# MAGIC %md
# MAGIC # VADER

# COMMAND ----------

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

# COMMAND ----------

df = pd.read_csv('tweets_cleaned.csv')
df

# COMMAND ----------

df.info()

# COMMAND ----------

df.dropna(inplace=True)

# COMMAND ----------

df.iloc[0]['tweet']

# COMMAND ----------

sid.polarity_scores(df.iloc[0]['tweet'])

# COMMAND ----------

df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# COMMAND ----------

df['scores_V'] = df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# COMMAND ----------

df.head()

# COMMAND ----------

df['compound'] = df['scores_V'].apply(lambda d: d['compound'])

# COMMAND ----------

df.head()

# COMMAND ----------

df['comp_vader'] = df['compound'].apply(lambda score: 'POSITIVE' if score >0 else ('NEUTRAL' if score ==0 else 'NEGATIVE'))

# COMMAND ----------

df.head()

# COMMAND ----------

df['comp_vader'].value_counts()

# COMMAND ----------

#plot a bar graph to show count of tweet sentiment
fig = plt.figure(figsize=(7,5))
color = ['green', 'grey', 'red']
df['comp_vader'].value_counts().plot(kind='bar',color = color)
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
tags=df['comp_vader'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, 
         startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of polarity')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # TextBlob

# COMMAND ----------

#get subjectivity and polarity of tweets with a function
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


#get polarity with a function
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df['Subjectivity'] = df['tweet'].apply(getSubjectivity)
df['Polarity'] = df['tweet'].apply(getPolarity)

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

 df['Analysis_TB'] = df['Polarity'].apply(getAnalysis)

# COMMAND ----------

df.head()

# COMMAND ----------

 df['Analysis_TB'].value_counts()

# COMMAND ----------

df['comp_vader'].value_counts()

# COMMAND ----------

#pie chart to show percentage distribution of polarity
fig = plt.figure(figsize=(7,7))
colors = ('green',  'grey', 'red')
wp={'linewidth':2, 'edgecolor': 'black'}
tags=df['Analysis_TB'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, 
         startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of polarity')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # FLAIR

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
df['scores_flair'] = df['tweet'].apply(lambda s: score_flair(s)[0])

# Predict sentiment label for each review
df['pred_flair'] = df['tweet'].apply(lambda s: score_flair(s)[1])

# Check the distribution of the score
#df['scores_flair'].describe()

# COMMAND ----------

# Check the distribution of the score
df['scores_flair'].describe()

# COMMAND ----------

df['pred_flair'].value_counts()

# COMMAND ----------

df.head(50)

# COMMAND ----------

#pie chart to show percentage distribution of polarity
fig = plt.figure(figsize=(7,7))
colors = ('green', 'red')
wp={'linewidth':2, 'edgecolor': 'black'}
tags=df['pred_flair'].value_counts()
explode = (0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, 
         startangle=90, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of polarity')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's compare the Negative situation

# COMMAND ----------

# record 2
df.iloc[2]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it looks like hidden negative feeling. 
# MAGIC - Flair gets the meaning (Negative) whereas both Vader and TextBlob get Postive meaning. Both of them fail to guess the meaning.

# COMMAND ----------

# record 3
df.iloc[3]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems taylor was blocked from twitter and Musk has reinstated her account.
# MAGIC The feeling is positive but what about who blocked Taylor and the story behind that.
# MAGIC - Flair gets the meaning (Negative) whereas both Vader and TextBlob get Neutral meaning. Both of them guess the meaning better than Flair. But what about if Elon Musk who gave the order to block Taylor? 

# COMMAND ----------

# record 7
df.iloc[7]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Negative
# MAGIC - Flair and Vader get the meaning (Negative) whereas TextBlob get Neutral meaning. TextBlob fails to guess the meaning.

# COMMAND ----------

# record 11
df.iloc[11]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Negative against Elon Musk and Trumps
# MAGIC - Flair gets the meaning (Negative) whereas both Vader and TextBlob get Positive. Both of them fail to guess the meaning.

# COMMAND ----------

# record 12
df.iloc[12]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Negative against Elon Musk.
# MAGIC - Vader gets the meaning (Negative) whereas both Flir gets Positive and TextBlob get Neutral. Both of them fail to guess the meaning.

# COMMAND ----------

# record 20
df.iloc[20]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Negative against Elon Musk. It is a kind of negative  Question d'exclamation!? 
# MAGIC - Flair gets the meaning (Negative) whereas both Vader gets Neutral and TextBlob get Positive. Both of them fail to guess the meaning.

# COMMAND ----------

# record 25
df.iloc[25]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Neutral
# MAGIC - TextBlob gets the meaning (Neutral) whereas both Flir gets Negative and Vader get Positive. Both of them fail to guess the meaning.

# COMMAND ----------

# record 28
df.iloc[28]['tweet']

# COMMAND ----------

# MAGIC %md
# MAGIC This expresion, it seems Negative against Elon Musk. 
# MAGIC - Flair gets the meaning (Negative) whereas both Vader and TextBlob get Positive. Both of them fail to guess the meaning.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Result
# MAGIC - Flair is more sentiment analysing emotion and opinion than Vader and TextBlod.
# MAGIC

# COMMAND ----------



# COMMAND ----------

df.columns

# COMMAND ----------



# COMMAND ----------

df_VTBF = df.drop(['tweet', 'scores_V', 'compound', 'Subjectivity', 'Polarity', 'scores_flair'], axis=1)
df_VTBF

# COMMAND ----------

df_VTBF.head()

# COMMAND ----------

ElonMusk_Sentiment = df.drop(['scores_V', 'compound', 'Subjectivity', 'Polarity', 'scores_flair', 'comp_vader', 'Analysis_TB'], axis=1)
ElonMusk_Sentiment

# COMMAND ----------

# MAGIC %md
# MAGIC we need to map the ‘NEGATIVE’ value to 0 and the ‘POSITIVE’ value to 1 

# COMMAND ----------

# Change the label of flair prediction to 0 if negative and 1 if positive
mapping = {'NEGATIVE': 0, 'POSITIVE': 1}
ElonMusk_Sentiment['pred_flair'] = ElonMusk_Sentiment['pred_flair'].map(mapping)

# COMMAND ----------

ElonMusk_Sentiment

# COMMAND ----------

ElonMusk_Sentiment = ElonMusk_Sentiment.rename(columns={'pred_flair':'Sentiment'})

# COMMAND ----------

ElonMusk_Sentiment

# COMMAND ----------

ElonMusk_Sentiment.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a word cloud for the tweets
# MAGIC
# MAGIC To understand which words have been used most in the tweets, we create a word cloud function for both positive and negative tweets.

# COMMAND ----------

#create a function for wordcloud

def create_wordcloud(text):    
    allWords = ' '.join([tweets for tweets in text])
    wordCloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(allWords)
    plt.figure(figsize=(20,10))
    plt.imshow(wordCloud)
    plt.axis('off')
    plt.show()
    

# COMMAND ----------

#wordcloud for positive tweets
posTweets = df.loc[df['pred_flair']=='POSITIVE', 'tweet']
create_wordcloud(posTweets)

# COMMAND ----------

#wordcloud for negative tweets
negTweets = df.loc[df['pred_flair']=='NEGATIVE', 'tweet']
create_wordcloud(negTweets)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Finding the most popular words in tweets and their frequency
# MAGIC
# MAGIC Here, every tweet is broken down into words and analyzed

# COMMAND ----------

#break each tweet sentence into words

sentences = []

for word in df['tweet']:
    sentences.append(word)
    
sentences

lines = list()

for line in sentences:
    words = line.split()
    for w in words:
        lines.append(w)
        
lines[:20] #shows first 10 words in the first tweet

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we remove stop words which are the common words used in the English Language such as ‘on’, ‘the’, ‘is’ etc. We then group the rest together to their root words eg joined, joining, and joint are grouped together as a single word — join and save it to a new data frame df.

# COMMAND ----------

#stemming all the words to their root word

stemmer = SnowballStemmer(language='english')

stem=[]

for word in lines:
    stem.append(stemmer.stem(word))
    
stem[:20]


#removes stopwords (very common words in a sentence)

stem2 = []

for word in stem:
    if word not in nlp.Defaults.stop_words:
        stem2.append(word)
        
        
#creates a new dataframe for the stem and shows the count of the most used words

df = pd.DataFrame(stem2)
df=df[0].value_counts()
df #shows the new dataframe


# COMMAND ----------

df.head(10)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Finally, we plot the most used words.

# COMMAND ----------

df.head(20).plot(kind='bar',title='Top Words', color = ['blue','red','green','yellow','orange'])
plt.xlabel('Count of Words')
plt.ylabel('Words from Twitter');

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

ElonMusk_Sentiment.to_csv('ElonMusk_Sentiment.csv', index=False)

# COMMAND ----------

pd.read_csv('ElonMusk_Sentiment.csv')

# COMMAND ----------


