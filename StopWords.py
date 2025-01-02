import nltk

nltk.download('punkt_tab')

TEXT = "I am a girl. I am studying btech then , i will not ? study . I like to eat and sleep"

from nltk import word_tokenize
words = word_tokenize(TEXT)
print("--- words count :" + str(len(words)) + "\n")


from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(string.punctuation)
print(stop_words)

cleanWords = [ w for w in words if not w in stop_words]
print("---  clean words :" + str(len(cleanWords)) + "\n")

i = 0
for word in cleanWords:
    i = i +1
    print("-" + str(i) + ":" + word )






