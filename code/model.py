import pandas as pd
import math
import numpy as np

from keras.layers import Embedding
from keras.layers import SpatialDropout1D, Dense, Input, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, LSTM
from keras.models import Model
from keras.layers import Bidirectional, TimeDistributed, Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate

from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import Model
#from keras_contrib.layers import CRF
from keras.models import load_model

print("Loading X...")
X = np.genfromtxt("./data/x_train.csv", delimiter=',') 

print("Loading embedding...")
embedding_matrix = np.genfromtxt("./data/embedding_matrix.csv")

print("Loading Y...")
Y = np.loadtxt("./data/y_train.csv", delimiter=',')

n_words = 404821
max_len = 75
n_tags = 48
EMBEDDING_DIM = 100
VOCAB_DIM = 269346

print("Reshaping X...")
X = X.reshape((200000, 75))

print("Reshaping emb...")
embedding_matrix = embedding_matrix.reshape((VOCAB_DIM, EMBEDDING_DIM))

print("Reshaping Y...")
Y = Y.reshape((200000, max_len, n_tags))

input = Input(shape=(max_len,))
model = Embedding(input_dim=VOCAB_DIM, output_dim=EMBEDDING_DIM,
                  input_length=max_len, weights=[embedding_matrix], mask_zero=True, trainable=True)(input)  # 100-dim embedding
model = Bidirectional(LSTM(units=100, return_sequences=True,
                           recurrent_dropout=0.1))(model)
out = Dense(n_tags, activation="softmax")(model)

model = Model(input, out)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model.summary()

# Fit 
model.fit(X, Y, batch_size=512, epochs=4, validation_split=0.1, verbose=1)

# Save
model.save('./models/model_lstm_100.h5')