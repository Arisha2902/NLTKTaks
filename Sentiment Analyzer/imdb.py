import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input
from tensorflow.keras.preprocessing import sequence
import numpy as np

# Disable OneDNN optimizations (if necessary for system compatibility)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load preprocessed IMDB dataset, split into training and testing sets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

# Build word index dictionary
word_to_id = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

# Pad sequences to normalize input length to 300
x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300)

# Build a sequential neural network
model = Sequential([
    Embedding(input_dim=5000, output_dim=32, input_length=300),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model for 3 epochs with a batch size of 64
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

# Evaluate the model performance on test data
result = model.evaluate(x_test, y_test, verbose=0)
print(f"Model Accuracy: {result[1]}")

# Testing example reviews
negative = "this movie is bad"
positive = "i had fun"
negative2 = "this movie was terrible"
positive2 = "i really liked the movie"


# Function to preprocess reviews for prediction
def preprocess_review(review):
    words = review.split(" ")
    encoded = [word_to_id.get(word, word_to_id["<UNK>"]) for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=300)
    return np.array(padded)


# Predict sentiment for the given reviews
for review in (positive, positive2, negative, negative2):
    input_data = preprocess_review(review)
    prediction = model.predict(input_data, verbose=0)[0][0]
    print(f"{review} -- Sentiment Score: {prediction}")
