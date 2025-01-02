import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('tagsets_json')
nltk.download('wordnet')

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
# print(taggedWords)
# print(nltk.help.upenn_tagset())
#for using above code " nltk.download('tagsets_json')"

#stemming
#in stemming some words may not belong to lamguage but in lemetization they all belong to language
# from nltk.stem import  LancasterStemmer
# stemmer = LancasterStemmer()

# snowball stemmer
# ye galat h,yha kuch shi krna h
# from nltk.stem import SnowballStemmer
# stemmer = SnowballStemmer()

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


i = 0
for word in cleanWords:
    i = i+1
    print("-" + str(i) + word + ":" + stemmer.stem(word))

# lemmetizing
# to do lemmetizing use " nltk.download('wordnet')"
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print("lemmetizing")
i = 0
for word in cleanWords:
    i = i+1
    print("-" + str(i) + word + ":" + lemmatizer.lemmatize(word))
    # print("-" + str(i) + word + ":" + lemmatizer.lemmatize(word,pos ="v"))



