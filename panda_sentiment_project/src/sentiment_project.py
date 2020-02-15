import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from os import getcwd, listdir
from string import punctuation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from scikitplot.metrics import plot_confusion_matrix


# Setting Stop Words
stop_words = set(stopwords.words('english'))
stop_words
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


## Establishing Snowball Stemmer
snowball = SnowballStemmer('english')


## Treebank to WordNet Tag Translator Function Borrowed from Suzana on StackOverFlow
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Getting Final, Lemmatized Sentence
new_sentence = "He plays the guitar very well."
preprocessed_sentence = preprocess(new_sentence)
table = str.maketrans('', '', punctuation)
stripped_sentence = [word.lower().translate(table) for word in preprocessed_sentence.split()]
stripped_sentence = " ".join([word for word in stripped_sentence if word not in stop_words])
tagged = pos_tag(word_tokenize(stripped_sentence))

print(tagged)

tagged = [i[1] for i in tagged]
tagged
type_tagged = [type(i) for i in tagged]
type_tagged
tagged = [get_wordnet_pos(tag) for tag in tagged]
tagged
lem_word_list = [lem.lemmatize(word, tagged[i]) for i, word in enumerate(stripped_sentence.split(" "))]
lem_word_list
stripped_sentence.split(" ")

## Removing Punctuation, Preprocessing, Lemmatizing and Tokenizing Tweets
def lem_tokenize(tweet):
    # Preprocessing Text
    tweet = preprocess(tweet)
    # Removing Punctuation
    table = str.maketrans('', '', punctuation)
    stripped_sentence = [word.lower().translate(table) for word in tweet.split()]
    stripped_sentence = " ".join([word for word in stripped_sentence if word not in stop_words])
    # Tokenizing the Words in the Sentence and Part of Speech Tagging
    tagged = pos_tag(word_tokenize(stripped_sentence))
    tagged = [i[1] for i in tagged]
    tagged = [get_wordnet_pos(tag) for tag in tagged]
    # Lemmatizing Each Word
    lem_word_list = [lem.lemmatize(word, tagged[i]) for i, word in enumerate(stripped_sentence.split(" "))]
    # Returning Final String
    return " ".join(lem_word_list)

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

sample_sentence = "Playing is a sport that I played since the beginning of the word play."

# Comparing Lemmatizing to Snowball Stemming

lem_tokenize(sample_sentence)
snow_tokenize(sample_sentence)

## Filtering Data Frame for Happy and Sad Sentiments Only
new_df = df[(df['sentiment'].str.contains('happiness')) | (df['sentiment'].str.contains('sadness'))]
new_df.columns = ['id', 'sentiment', 'tweet']
tweet_list = new_df['tweet'].tolist()

snowball_tweets = [snow_tokenize(tweet) for tweet in new_df['tweet'].tolist()]
clean_df = new_df.copy()

clean_df['snowball_stemmed_tweets'] = snowball_tweets

clean_df.head()











# Leaving the below code for Lemmatizing Purposes in the Future

# Function to remove mystery cases where wordnet POS tag and stripped sentence lengths don't match

def mystery_cases(tweet_list: list) -> list:
    # Empty List to Store All the Mystery Cases
    mys_cases = []
    # Taking Each Tweet and Comparing The Lengths of Their WordNet POS Tags and Processed Sentence Lengths
    for i,tweet in enumerate(tweet_list):
        table = str.maketrans('', '', punctuation)
        stripped_tweet = [word.lower().translate(table) for word in tweet.split()]
        stripped_tweet = " ".join([word for word in stripped_tweet if word not in stop_words])
        tagged = pos_tag(word_tokenize(stripped_tweet))
        tagged = [i[1] for i in tagged]
        if len(tagged) != len(stripped_tweet.split()):
            mys_cases.append((i, tweet))
    return mys_cases

# Storing the mystery cases in a list and removing them from the data frame
mys_cases = mystery_cases(tweet_list)
type(mys_cases)
len(mys_cases)
mys_cases
new_df.shape[0] - len(mys_cases)
mys_indices = [i[0] for i in mys_cases]
clean_df = new_df.copy()
clean_df.dropna(inplace=True)
clean_df.shape
clean_df['sentiment'].value_counts()

mystery_cases = clean_df.iloc[mys_indices]
final_df = clean_df[clean_df['id'].isin(mystery_cases['id']) == False]
final_df.shape
