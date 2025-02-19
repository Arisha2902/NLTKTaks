# from cgi import maxlen

import os

from tensorflow.python.keras.saving.saved_model.load import metrics
from tensorflow.python.ops.math_ops import negative
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

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


# preprocessed data, seprated in traning and testing
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 5000)

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = { k:(v+3) for k,v in word_to_id.items() }
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# normalize the data ,and limit it
x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test =sequence.pad_sequences(x_test,maxlen=300)


# build a model network
# added 3 laywrs to it flatten, embedding, dense
network = Sequential()
network.add(Embedding(5000, 32))
network.add(Flatten())
network.add(Dense(1, activation='sigmoid'))
# compile the network
network.compile(loss="binary_crossentropy", optimizer='Adam',metrics=['accuracy'])

# train the model
network.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=3, batch_size=64)

# to chrck the rsult using 2 varaiable x_test, y_test
result = network.evaluate(x_test , y_test, verbose = 0)


negative = "this movie is bad"
positive = "i had fun"
negative2 = "this movie was terrible"
positive2 = "i really liked the movie"
#
for review in (positive, positive2, negative , negative2):
    temp = []
    for word in review.split(" "):
        temp.append(word_to_id[word])
    temp_padded = sequence.pad_sequences([temp], maxlen = 300)
    print(review + " -- Sent --" + str(network.predict(array([temp_padded][0]))[0][0]))


