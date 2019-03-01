import sys
from Vocabulary import Vocabulary
from Reader import Reader
import pickle
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('-d', type=int, default=150, dest='dimensionality', help='Trained embedding dimensionality')
parser.add_argument('-e', type=int, default=5, dest='epochs', help='Trained embedding dimensionality')
parser.add_argument('-c', type=int, default=200, dest='context', help='Number of contexts in batch')
parser.add_argument('-n', type=int, default=5, dest='negative', help='Number of negative samples')
parser.add_argument('-w', type=int, default=5, dest='window_size', help='Context window size (on one side)')
parser.add_argument('-b', type=int, default=1000, dest='batch_size', help='Training batch size')
parser.add_argument('-m', type=str, default='skipgram', dest='model_name', help='Trained model')
parser.add_argument('-s', type=float, default=1e-4, dest='subsampling_parameter', help='Subsampling threshold')
parser.add_argument('-l', type=str, default='en', dest='language', help='Language of wikipedia dump')
parser.add_argument('-sgm', type=str, dest='segmenter', help='Segmenter Path')
parser.add_argument('-wiki', type=bool, default=False, dest='wiki', help='Read from wikipedia dump')
parser.add_argument('data_path', type=str, help='Path to training data. Can be plain file or wikipedia dump. Set flag \'--wiki\' if using wiki dump')
parser.add_argument('voc_path', type=str, help='Path to vocabulary dump')


args = parser.parse_args()
n_dims = args.dimensionality
epochs = args.epochs
n_contexts = args.context
k = args.negative
window_size = args.window_size
model_name = args.model_name
data_path = args.data_path
vocabulary_path = args.voc_path
wiki = args.wiki
lang = args.language
sgm_path = args.segmenter

graph_saving_path = "./models/%s" % model_name
ckpt_path = "%s/model.ckpt" % graph_saving_path

vocab_progressions = [10000, 20000, 50000, 100000, 200000]

voc = pickle.load(open(vocabulary_path, "rb"))

voc.set_subsampling_param(1e-4)

reader = Reader(data_path, voc, n_contexts, window_size, k, wiki=wiki, lang=lang)

# if model_name != 'skipgram':
#     segmenter = WordSegmenter(sgm_path, lang)
#     sgm = lambda x: segmenter.segment(x, str_type=True)
#
#     def next_batch(from_top_n=None):
#         pos, neg, lbl = reader.next_batch(from_top_n=from_top_n)
#         return sgm(pos), sgm(neg), lbl
#
#     restore = segmenter.to_segments
#
# else:
#     def next_batch(from_top_n=None):
#         return reader.next_batch(from_top_n=from_top_n)
#
#     restore = lambda x: voc.id2word[x]

def next_batch(from_top_n=None):
    return reader.next_batch(from_top_n=from_top_n)

arg_dict = vars(args)
arg_dict['full_vocabulary_size'] = len(voc)
arg_dict['graph_saving_path'] = graph_saving_path
arg_dict['ckpt_path'] = ckpt_path
print(len(arg_dict))
for key, val in arg_dict.items():
    print(f"{key}={val}")

def seld_line(a, p, l):
    print("%d\t%d\t%d" % (a, p, l))
    # if model_name != 'skipgram':
    #     # t1 = " ".join(map(str,a))
    #     # t2 = " ".join(map(str,p))
    #     # t1 = " ".join([str(el) for el in a])
    #     # t2 = " ".join([str(el) for el in p])
    #     print("%s\t%s\t%d" % (a, p, l))
    #     # sys.stdout.write("%s\t%s\t%d\n" % (a, p, l))
    #
    # else:
    #     print("%d\t%d\t%d" % (a, p, l))
    #     # sys.stdout.write("%s\t%s\t%d\n" % (a, p, l))

for vocab_size in vocab_progressions:

    print("vocab=%d" % vocab_size)

    for e in range(epochs):

        print("epoch=%d" % e)
        batch = next_batch(from_top_n=vocab_size)

        while batch is not None:
            for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
                seld_line(a, p, l)
                # pass
            batch = next_batch(from_top_n=vocab_size)
            # print("boom")

    epochs += 2