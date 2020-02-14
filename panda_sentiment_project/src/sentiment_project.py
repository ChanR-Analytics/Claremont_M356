import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import getcwd, listdir
from hate_speech_project.src.python.hate_speech_oop import tokenize, HatebaseTwitter

df = pd.read_csv("https://query.data.world/s/33zqxxshjwehd5xoktqpxyiuyqzedm")
df.drop("author", axis=1, inplace=True)

# Visualizing the Sentiments
sentiment = df['sentiment']
x = sentiment.value_counts()
x.sort_index()
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
