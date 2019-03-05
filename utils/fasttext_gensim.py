from gensim.models import FastText
import time
import os
import logging
from Tokenizer import Tokenizer


class MySentences(object):
    def __init__(self, fname):
        self.dirname = fname
        self.tok = Tokenizer()
 
    def __iter__(self):
        for line in open(self.dirname):
            yield self.tok(line, lower=True, hyphen=False)



# parameters
dim = 150
window = 5
min_count = 2
negative = 20
sample = 1e-4
iter_ = 1
alpha = 0.001
min_alpha = 0.001
max_vocab_size = 100000

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = MySentences('wiki_plain.txt')

print("Word2vec training started, ", time.strftime("%Y-%m-%d %H:%M"))
model = FastText(sentences, size=300, window=5, min_count=2, workers=4, sg=1, negative=20, ns_exponent=0.75,
                 sample=1e-4, iter=1, alpha=0.05, min_alpha=5e-3, sorted_vocab=1, max_vocab_size=250000, word_ngrams=1, bucket=120000, min_n=3, max_n=4)

print("Word2vec training finished, ", time.strftime("%Y-%m-%d %H:%M"))


model.wv.save_word2vec_format('dumped_ft/' + 'emb.txt')
print("Embeddings saved, ", time.strftime("%Y-%m-%d %H:%M"))
model.save('dumped_ft/' + 'model')
print("Model saved, ", time.strftime("%Y-%m-%d %H:%M"))
