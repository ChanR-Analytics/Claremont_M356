import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import getcwd, listdir
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from xgboost import XGBClassifier
from scikitplot.metrics import plot_confusion_matrix
from hate_speech_project.src.python.hate_speech_oop import tokenize, HatebaseTwitter


hb_path = getcwd() + "/hate_speech_project/data/HatebaseTwitter"
hb = HatebaseTwitter(hb_path)
features = hb.features()


df = pd.read_csv("https://query.data.world/s/33zqxxshjwehd5xoktqpxyiuyqzedm")
df.drop("author", axis=1, inplace=True)
df.head()

print(tokenize(hb.df['tweet'].tolist()[0]))  

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

# Getting the Dataset with Final Features
csv_path = getcwd() + "/panda_sentiment_project/data"
final_df = pd.read_csv(f"{csv_path}/{listdir(csv_path)[0]}")
final_df.head()
# Machine Learning
X = final_df[['compound', 'negative', 'neutral', 'positive']]
y = final_df['sentiment']
y.replace("happiness", 1, inplace=True)
y.replace("sadness", 0, inplace=True)
X.shape
y.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train.shape
Y_train.shape

X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test).reshape(-1, 1)
model = LogisticRegression(solver='saga',l1_ratio=0.1, penalty='elasticnet')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy_score(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))

rf_model = RandomForestClassifier(n_estimators=600, max_depth=5, bootstrap=True)
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)

accuracy_score(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))

ada_model = AdaBoostClassifier(n_estimators=400, learning_rate=0.000005)
ada_model.fit(X_train, Y_train)
Y_pred = ada_model.predict(X_test)

accuracy_score(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))


xgb_model = XGBClassifier(n_estimators=400, learning_rate=0.00000005)
xgb_model.fit(X_train, Y_train)

Y_pred = xgb_model.predict(np.array(X_test))

accuracy_score(Y_pred, Y_test)
print(classification_report(Y_pred, Y_test))


plt.figure(figsize=(10,10))
plot_confusion_matrix(Y_test, Y_pred)
