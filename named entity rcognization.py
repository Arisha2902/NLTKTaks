import nltk

from Tokenize import sentence

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('tagsets_json')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

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
taggedWords = pos_tag(cleanWords)


#stemming
from nltk.stem import  LancasterStemmer
# stemmer = LancasterStemmer()
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()


i = 0
for word in cleanWords:
    i = i+1
    # print("-" + str(i) + word + ":" + stemmer.stem(word))

# lemmetizing
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# NER
from nltk import ne_chunk, sent_tokenize

sentences = sent_tokenize(TEXT)
# to use this "nltk.download('maxent_ne_chunker_tab')" and "nltk.download('words')"
for sentence in sentences:
    words = word_tokenize(sentence)
    tags = pos_tag(words)
    ner = ne_chunk(tags)
    ner.draw()
 # cnnxx