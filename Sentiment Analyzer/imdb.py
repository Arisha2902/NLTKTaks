import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300)



