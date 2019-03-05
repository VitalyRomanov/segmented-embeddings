from gensim.models import Word2Vec
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
min_count = 5
negative = 20
sample = 1e-4
iter_ = 1
alpha = 0.025
min_alpha = 0.001
max_vocab_size = 100000

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = MySentences('wiki_plain.txt')

print("Word2vec training started, ", time.strftime("%Y-%m-%d %H:%M"))
model = Word2Vec(sentences, size=dim, window=window, min_count=min_count, workers=4, sg=1, negative=negative, ns_exponent=0.75,
                 sample=sample, iter=iter_, compute_loss=True, alpha=alpha, min_alpha=min_alpha, max_vocab_size=max_vocab_size, sorted_vocab=1)

print("Word2vec training finished, ", time.strftime("%Y-%m-%d %H:%M"))


model.wv.save_word2vec_format('dumped/' + 'emb.txt')
print("Embeddings saved, ", time.strftime("%Y-%m-%d %H:%M"))
model.save('dumped/' + 'model')
print("Model saved, ", time.strftime("%Y-%m-%d %H:%M"))