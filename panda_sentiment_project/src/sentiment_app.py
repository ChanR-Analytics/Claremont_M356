import numpy as np
import pandas as pd
from os import getcwd, listdir
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef

stopwords = nltk.corpus.stopwords.words('english')

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text

## Establishing Snowball Stemmer
snowball = SnowballStemmer('english')

## Removing Punctuation, Preprocessing, Lemmatizing and Tokenizing Tweets
def sent_preprocess(tweet):
    # Preprocessing Text
    tweet = preprocess(tweet)
    # Removing Punctuation
    table =str.maketrans('', '', punctuation)
    stripped_sentence = [word.lower().translate(table) for word in tweet.split()]
    stripped_sentence = " ".join([word for word in stripped_sentence if word not in stop_words])
    return stripped_sentence

def snow_tokenize(tweet):
    # Preprocessing Text
    tweet = preprocess(tweet)
    # Removing punctuation
    table = str.maketrans('', '', punctuation)
    stripped_sentence = [word.lower().translate(table) for word in tweet.split()]
    stripped_sentence = " ".join([word for word in stripped_sentence if word not in stop_words])
    # Tokenizing the Words in the Sentence
    tokenized = word_tokenize(stripped_sentence)
    # Stemming Each Word in the Sentence
    stemmed_words = [snowball.stem(word) for word in tokenized]
    return " ".join(stemmed_words)


# Getting Sentiment Scores for Both Preprocessed Sentence Tweets and Snowball Stemmed Tweets
analyzer = SentimentIntensityAnalyzer()


# Creating the First Part of the App
st.write("# ChanR Analytics Presents: Panda Sentiment Analysis")
data_path = getcwd() + "/panda_sentiment_project/data/happy_sad_sentiment_data.csv"
df = pd.read_csv(data_path)
st.write("## Section 1: Exploring the Data")
img_path = getcwd() + "/panda_sentiment_project/data_viz"
st.image(f"{img_path}/{listdir(img_path)[0]}")
st.write(df['sentiment'].value_counts())
st.write(df.head())

# Machine Learning Component


# Wrapping Images to Predictions
