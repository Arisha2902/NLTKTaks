import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('tagsets_json')

TEXT = "I am a girl. I am studying btech then , i will not ? study . I like to eat and sleep"

from nltk import word_tokenize
words = word_tokenize(TEXT)


from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(string.punctuation)
cleanWords = [ w for w in words if not w in stop_words]

#pos
from nltk import pos_tag
#for using above "nltk.download('averaged_perceptron_tagger_eng')"
taggedWords = pos_tag(cleanWords)
print(taggedWords)
print(nltk.help.upenn_tagset())
#for using above code " nltk.download('tagsets_json')"

#stemming
from nltk.stem.lancaster import  LancasterStemmer
# from nltk.ste..por
stemmer = LancasterStemmer


i = 0
for taggedWord in taggedWords:
    i = i+1
    print("-" + str(i) + ":" + str(taggedWord))