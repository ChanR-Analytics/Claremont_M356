import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from os import getcwd, listdir
from string import punctuation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from scikitplot.metrics import plot_confusion_matrix


# Borrowing Davidson's Preprocess() Function
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

# Reading in the Sentiments Data Frame Annotated by CrowdFlower on Data.World
df = pd.read_csv("https://query.data.world/s/33zqxxshjwehd5xoktqpxyiuyqzedm")
df.drop("author", axis=1, inplace=True)
df.head()

# Visualizing the Sentiment Counts
sentiment = df['sentiment']
x = sentiment.value_counts()
print(x)

img_path = getcwd() + "/panda_sentiment_project/data_viz"


x=x.sort_index()
plt.figure(figsize=(10,6))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Sentiment Count", size=18)
plt.ylabel('Count', size=14)
plt.xlabel('Sentiments', size=14)
labels = x.values
plt.savefig(f"{img_path}/sentiment_counts.png")

# Preparing the NLP Features

## If punkt and wordnet already installed, feel to comment out lines 61 and 62.
nltk.download('punkt')
nltk.download('wordnet')

## Establishing Lemmatizer
lem = WordNetLemmatizer() 

## Removing Punctuation, Preprocessing, Lemmatizing and Tokenizing Tweets
def tokenize(tweet):
    # Preprocessing Text
    tweet = preprocess(tweet)
    # Removing Punctuation
    word_list = tweet.split()
    table = str.maketrans('', '', punctuation)
    stripped_word_list = [word.translate(table) for word in word_list]
    # Lemmatizing Each Word
    lem_word_list = [lem.lemmatize(word) for word in stripped_word_list]
    # Tokenizing the Sentence
    sentence = " ".join(lem_word_list)
    return " ".join(sent_tokenize(sentence))

sample_sentence = "Hello, my name is Rishov!"

tokenize(sample_sentence)
