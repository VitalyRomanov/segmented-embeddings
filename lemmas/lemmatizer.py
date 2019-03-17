import sys
import pickle
voc_path = sys.argv[1]
lang = sys.argv[2]

from spacy.lang.en import English
from spacy.lang.ru import Russian

if lang == 'en':
    import spacy
    model = spacy.load(lang, disable=['parser', 'tagger','ner'])
    nlp = lambda x: model(x)[0].lemma_
else:
    import pymorphy2
    model = pymorphy2.MorphAnalyzer()
    nlp = lambda x: model.parse(x)[0].normal_form

with open(voc_path, "r") as voc_file:
    words = voc_file.read().strip().split("\n")

word2id = dict(zip(words, range(len(words))))

# nlp = spacy.load(lang, disable=['parser', 'tagger','ner'])

word_lemma = [(word, str(nlp(word))) for word in words]

_, lemmas = zip(*word_lemma)
lemmas = list(set(lemmas))
lemmas.sort()

lemma2id = dict(zip(lemmas, range(len(lemmas))))

w2l = dict()

for word, lemma in word_lemma:
   w2l[word2id[word]] = [lemma2id[lemma]] 

print(len(lemma2id))

pickle.dump(w2l, open("word2segment.pkl", "wb"))
pickle.dump(lemma2id, open("segment2id.pkl", "wb"))




with open("lemmas.txt", "w") as lemmas_out:
    for word, lemma in word_lemma:
        lemmas_out.write("%s\t%s\n" % (word, lemma))
