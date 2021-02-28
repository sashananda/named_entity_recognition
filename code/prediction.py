import math
import numpy as np
from keras.models import load_model
import pandas as pd
import string

from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

print("Loading keras model...")
model = load_model('./models/model_lstm_100.h5')

print("Loading X...")
X = np.genfromtxt("./data/x_test.csv", delimiter=',') 
print("Loading Y...")
Y = np.loadtxt("./data/y_test.csv", delimiter=',')

max_len = 75
n_tags = 48

print("Reshaping X...")
X = X.reshape((563, 75))

print("Reshaping Y...")
Y = Y.reshape((563, max_len, n_tags))

# Predict
Y_pred = model.predict(X, verbose=1)

# Round to one hot
Y_pred_one_hot = []
prob = 0.3
for i in range(len(Y_pred)):
    pred = np.where(Y_pred[i] > prob, 1, 0)
    Y_pred_one_hot.append(pred)

Y_pred_one_hot = np.array(Y_pred_one_hot)

mlb = MultiLabelBinarizer()
classes = ['/art', '/astral_body', '/award', '/biology', '/body_part',
       '/broadcast', '/broadcast_network', '/broadcast_program',
       '/building', '/chemistry', '/computer', '/disease', '/education',
       '/event', '/finance', '/food', '/game', '/geography', '/god',
       '/government', '/government_agency', '/internet', '/language',
       '/law', '/living_thing', '/livingthing', '/location', '/medicine',
       '/metropolitan_transit', '/military', '/music', '/news_agency', '/none',
       '/organization', '/park', '/people', '/person', '/play',
       '/product', '/rail', '/religion', '/software', '/time', '/title',
       '/train', '/transit', '/transportation', '/written_work']
mlb.fit([classes])

# Get tokenized array in order to get sentence structure
# Load the data
print("Loading data...")
df_train = pd.read_json('./data/train.json')
df_test = pd.read_json('./data/test.json')
df_train = df_train.loc[:199999, :]
df = pd.concat([df_train, df_test], ignore_index=True)

print("Tokenizing...")
sentences = df['sent'].tolist()
tokenizer = MWETokenizer()
tokenized_list = [tokenizer.tokenize(sentence.split()) for sentence in sentences]

# Remove punctuation from tokenized data
stop = list(string.punctuation)
# Remove stop words
tokenized = []
words = []
for sentence in tokenized_list:
    lst = []
    for i in sentence:
        if i not in stop:
            lst.append(i)
            words.append(i)
    tokenized.append(lst)

tokenized_test = tokenized[200000:]
y_pred_sent = []
y_true_sent = []

print("Getting predictions by sentence...")
for i in range(Y_pred_one_hot.shape[0]):
    N = len(tokenized_test[i])
    pred = Y_pred_one_hot[i][:N]
    for word in pred:
        all_zeros = np.count_nonzero(word)
        if (all_zeros == 0):
            word[32] = 1
            y_pred_sent.append(word)
        else:
            y_pred_sent.append(word)
    true = Y[i][:N]
    for word in true:
        y_true_sent.append(word)

y_pred_sent = np.array(y_pred_sent)
y_true_sent = np.array(y_true_sent)

print("F1 score: %f", f1_score(np.array(y_true_sent), np.array(y_pred_sent), average='weighted', zero_division=0))
print(classification_report(np.array(y_true_sent), np.array(y_pred_sent), target_names=list(mlb.classes_), zero_division=0))