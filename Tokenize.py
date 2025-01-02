import nltk

nltk.download('punkt_tab')

TEXT = "I am a girl. I am studying btech then i will not study.I like to eat and sleep"

from nltk import sent_tokenize

sentences = sent_tokenize(TEXT)

print("----Original Text :\n" + TEXT +"\n\n")

print("--- Sentences :" + str(len(sentences)) + "\n")
i = 0
for sentence in sentences:
    i = i +1
    print("-" + str(i) + ":" + sentence)

from nltk import word_tokenize

words = word_tokenize(TEXT)

print("----Original Text :\n" + TEXT +"\n\n")
print("--- words :" + str(len(words)) + "\n")
i = 0
for word in words:
    i = i +1
    print("-" + str(i) + ":" + word)