# Imports
import json
import pandas as pd
import math
import numpy as np
import string
from multiprocessing import cpu_count

from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.utils import simple_preprocess

from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from progress.bar import Bar

# Load the data
print("Loading data...")
df_train = pd.read_json('./data/train.json')
df_test = pd.read_json('./data/test.json')

# Choose a subset of the data for runtime purposes
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

# Train Word2Vec model on tokenized data
print("Training Word2Vec model...")
model = Word2Vec(sentences=tokenized, window=5, min_count=1, workers=4)

# Save the model 
print("Saving Word2Vec model...")
model.save("./word2vec.model")

# Get/set Word2Vec model parameters
EMBEDDING_DIM = 100
VOCAB_DIM = len(model.wv.vocab) + 1
vocab = list(model.wv.vocab)
vocab.append('ENDPAD')

# Get X (train and test combined)
print("Getting X...")
max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(list(set(words)))}
X = [[word2idx[w] for w in s] for s in tokenized]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

# Split into train and test and save
print("Saving X...")
X_train = X[:200000]
np.savetxt('./data/x_train.csv', X_train, delimiter=",")

X_test = X[200000:]
np.savetxt('./data/x_test.csv', X_test, delimiter=",")

# Get Y

# Get all unique classification labels
unique_labels = df['labels'].apply(tuple).unique()
flattened_unique_labels = list(set(list(sum(unique_labels, ()))))

# Get first level (root node) labels
root_mask = [label.count('/') == 1 for label in flattened_unique_labels]
root_level_labels = np.array(flattened_unique_labels)[root_mask]

leaf_mask = [label.count('/') == 2 for label in flattened_unique_labels]
leaf_level_labels = np.array(flattened_unique_labels)[leaf_mask]

# Add a root labels column to the df which only includes the root labels
def get_root_labels(labels):
    root_mask = [label.count('/') == 1 for label in labels]
    root_level_labels = np.array(labels)[root_mask]
    return list(root_level_labels)

df['root_labels'] = df['labels'].apply(get_root_labels)

# Get all the named entities for each sentence
def get_named_entities(sent, ents):
    named_entities = [sent[ent[1]:ent[2]] for ent in list(ents)]
    ents_split = []
    for ent in named_entities:
        ent_split = tokenizer.tokenize(ent.split())
        for ent_sp in ent_split:
            ents_split.append(ent_sp)
    return ents_split

df['named_entities'] = df.apply(lambda x: get_named_entities(x['sent'], x['ents']), axis=1)

# Create one-hot encoded columns for the root labels, and create y_train for the
# root multilabel classification
mlb = MultiLabelBinarizer()
y_train_root = mlb.fit_transform(df['root_labels'])
df_one_hot = pd.DataFrame(y_train_root, columns=list(mlb.classes_))

y_train_tags = []
y_train_tags_by_sentence = []

with Bar('Getting tags...') as bar:
    for i, row in df.iterrows():
        tags = []
        for word in tokenized[i]:
            if word in row['named_entities']:
                y_train_tags.append(row['root_labels'])
                tags.append(row['root_labels'])
            else:
                y_train_tags.append(['/none'])
                tags.append(['/none'])
        y_train_tags_by_sentence.append(tags)
        bar.next()

# Add none class to thee mlb
root_labels = list(mlb.classes_)
root_labels.append('/none')
mlb_2 = MultiLabelBinarizer()
mlb_2.fit([root_labels])

# Get Y with padding
max_len = 75
n_tags = 48
y_tags_one_hot = []
with Bar("One hot encoding tags...") as bar:
    for i in range(len(y_train_tags_by_sentence)):
        y_train_tag = y_train_tags_by_sentence[i]
        N = len(y_train_tag)
        one_hot = np.array(mlb_2.transform(y_train_tag))
        # Pad the array
        padding = np.zeros((max_len - N, n_tags - 1))
        # Add the none entry to the vector
        padding = np.insert(padding, 32, values=1, axis=1) 
        y_train_tag = np.concatenate((one_hot, padding), axis=0)
        y_tags_one_hot.append(y_train_tag)
        bar.next()

Y = np.array(y_tags_one_hot)

# Split into train and test and save
Y = Y.reshape(Y.shape[0], -1)
Y_test = Y[200000:].astype(int)
print("Saving Y test...")
np.savetxt('./data/y_test.csv', Y_test.astype(int), fmt='%i', delimiter=",")

print("Saving Y train...")
Y_train = Y[:200000].astype(int)
np.savetxt('./data/y_train.csv', Y_train.astype(int), fmt='%i', delimiter=",")

# Get word embedding matrix
print("Getting embedding matrix...")
embedding_matrix = np.zeros((VOCAB_DIM, EMBEDDING_DIM))
for i in range(len(vocab)):
    try:
        embedding_vector = model.wv[vocab[i]]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        vec = np.zeros(EMBEDDING_DIM)
        embedding_matrix[i] = vec

print("Saving embedding matrix...")
np.savetxt('./data/embedding_matrix.csv', embedding_matrix)