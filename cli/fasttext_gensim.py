from gensim.models import FastText
import time
import os
import logging
from utils.Tokenizer import Tokenizer
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('input_file', type=str, default=150, help='Path to text file')
parser.add_argument('output_dir', type=str, default=5, help='Ouput saving directory')
args = parser.parse_args()


class MySentences(object):
    def __init__(self, fname):
        self.dirname = fname
        self.tok = Tokenizer()
 
    def __iter__(self):
        for line in open(self.dirname):
            yield self.tok(line, lower=True, hyphen=False)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = MySentences(args.input_file)

print("FastText training started, ", time.strftime("%Y-%m-%d %H:%M"))
model = FastText(sentences,
                 size=300,
                 window=5,
                 min_count=2,
                 workers=4,
                 sg=1,
                 negative=20,
                 ns_exponent=0.75,
                 sample=1e-4,
                 iter=1,
                 alpha=0.05,
                 min_alpha=5e-3,
                 sorted_vocab=1,
                 max_vocab_size=100000,
                 word_ngrams=1,
                 bucket=120000,
                 min_n=3,
                 max_n=4)

print("FastText training finished, ", time.strftime("%Y-%m-%d %H:%M"))

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

model.wv.save_word2vec_format(args.output_dir + "/" + 'emb.txt')
print("Embeddings saved, ", time.strftime("%Y-%m-%d %H:%M"))
model.save(args.output_dir + "/" + 'model')
print("Model saved, ", time.strftime("%Y-%m-%d %H:%M"))
