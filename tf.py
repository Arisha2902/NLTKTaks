# TF : Term Frequency
# IDF: Inverse Document Frequency
# measure how important a word to a document in a collection of document
from dataclasses import field
# TF = ( no. of times the term appears in a document)/( total no. of terms in the document)
# IDF = log(total no. of documents/no. of documents with word in it)
# TFIDF = TF * IDF
# import  nltk
# nltk.download('averaged_perceptron_tagger_eng')
#
from math import log
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from nltk import corpus
import numpy

# importing corpus into data variable
data = corpus.brown
stemmer = PorterStemmer()

# building list of stopwords
stopwords = set(stopwords.words('english'))
stopwords =  stopwords.union(string.punctuation)

fileids = data.fileids()[:30]

idf_matrix = []
dictionary = dict()

# total count of words in corpus
words_count = 0

# total count of documents in corpus
documents_count = len(fileids)

# hold the words before and after filtering:stemming
filtered = dict()

# to save total count of every word per file
frequencies = dict()
for fileid in fileids:
    frequencies[fileid] = dict()

# filtering corpus
for fileid in fileids:
    for word in data.words(fileids):
        # skipping if it is a stop word
        if word is stopwords:
            continue

        before_word = word
        if before_word in filtered:
            word = filtered[before_word]
        else:
            word = stemmer.stem(before_word)
            filtered[before_word] = word

        if word in frequencies[fileid]:
            frequencies[fileid][word] += 1
        else:
            frequencies[fileid][word] = 1

        # saving all word in dictionery
        if word not in dictionary:
            dictionary[word] = words_count
            words_count += 1

# calculting TF
# index of non zero values
tf_matrix = []
nonzeroes = []
for fileid in fileids:
    tf_vector = [0] * words_count
    nonzeros_vec = []
    for word in frequencies[fileid].keys():
        index = dictionary[word]
        tf_vector[index] = frequencies[fileid][word]
        nonzeros_vec.append(index)
    nonzeroes.append(nonzeros_vec)
    tf_matrix.append(tf_vector)

# calculating IDF
idf_matrix = [0] * words_count
for fileid in fileids:
    for word in frequencies[fileid].keys():
        idf_matrix[dictionary[word]] += 1

# calculating TF-IDF matrix
tfidf = []

for i in range(documents_count):
    vector = [0] * words_count
    for j in nonzeroes[i]:
        tf_value = tf_matrix[i][j]
        idf_value = idf_matrix[j]
        tf_value = 1 + log(1 + documents_count/ float(idf_value),2)
        vector[j] = tf_value * idf_value
    tfidf.append(vector)

print(" ---- Top 10 : Keywords per documnet ---")
for i in range(len(tfidf)):
    print("---- Document : " + str(fileids[i]))
    vector = tfidf[i]
    sorted = numpy.argsort(vector)[:: -1]
    for ind in sorted[: 15]:
        print(ind)
