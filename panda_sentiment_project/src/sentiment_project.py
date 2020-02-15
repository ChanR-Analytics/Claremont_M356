import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
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


## Removing Punctuation, Preprocessing, Lemmatizing and Tokenizing Tweets
def tokenize(tweet):
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

sample_sentence = "Playing is a sport that I played since the beginning of the word play."

tokenize(sample_sentence)

## Filtering Data Frame for Happy and Sad Sentiments Only
new_df = df[(df['sentiment'].str.contains('happiness')) | (df['sentiment'].str.contains('sadness'))]
new_df.columns = ['id', 'sentiment', 'tweet']
tweet_list = new_df['tweet'].tolist()
tweet_list_one = tweet_list[:20]
tweet_list_two = tweet_list[21:24]
mystery_cases = [tweet_list[20], tweet_list[25]]
mystery_cases

# Preprocessing Text
tweet = preprocess(mystery_cases[0])
# Removing Punctuation
table = str.maketrans('', '', punctuation)
stripped_sentence = [word.lower().translate(table) for word in tweet.split()]
stripped_sentence = " ".join([word for word in stripped_sentence if word not in stop_words])
# Tokenizing the Words in the Sentence and Part of Speech Tagging
tagged = pos_tag(word_tokenize(stripped_sentence))
tagged = [i[1] for i in tagged]
tagged
tagged = [get_wordnet_pos(tag) for tag in tagged]
len(tagged)
len(stripped_sentence.split(" "))

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
mys_cases = mystery_cases(new_df['tweet'].tolist())
type(mys_cases)
len(mys_cases)
mys_cases
new_df.shape[0] - len(mys_cases)
mys_indices = [i[0] for i in mys_cases]

## Lemmatize Versus Normal Sentence Polarity Experiment

analyzer = SentimentIntensityAnalyzer()

example_tweet = new_df['tweet'].T[6]
example_tweet

def punctuation_process(text):
    processed_text = preprocess(text)
    clean_text = [word.lower().translate(table) for word in processed_text.split()]
    return " ".join(clean_text)

norm_text = punctuation_process(example_tweet)
tokenized_text = tokenize(example_tweet)

norm_results = analyzer.polarity_scores(norm_text)
tokenized_results = analyzer.polarity_scores(tokenized_text)

norm_results
tokenized_results
