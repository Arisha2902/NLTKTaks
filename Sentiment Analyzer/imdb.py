# from cgi import maxlen

import os

from tensorflow.python.keras.saving.saved_model.load import metrics

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from keras.datasets import imdb
import keras
from keras.models import Sequential
from tensorflow.keras.layers import Embedding

# from keras.layers.embeddings import Embedding
from keras.layers import Flatten , Dense
from keras.preprocessing import sequence
from numpy import array
# from tensorflow.keras.layers import Embedding



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 5000)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = { k:(v+3) for k,v in word_to_id.items() }
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test =sequence.pad_sequences(x_test,maxlen=300)

network = Sequential()
network.add(Embedding(5000, 32, input_length=300))
network.add(Flatten())
network.add(Dense(1, actvation='sigmoid'))
network.compile(loss="binary_crossentropy", optmizer='Adam',metrics=['accuracy'])

network.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=3, batch_size=64)

