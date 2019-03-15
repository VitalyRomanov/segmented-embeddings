import sys
from Vocabulary import Vocabulary
from Reader import Reader
import pickle
from aux import parse_args

args = parse_args()


# n_dims = args.dimensionality
# epochs = args.epochs
# n_contexts = args.context
# k = args.negative
# window_size = args.window_size
# model_name = args.model_name
# data_path = args.data_path
# vocabulary_path = args.voc_path
# wiki = args.wiki
# lang = args.language
# sgm_path = args.segmenter
# vocab_size = args.vocabulary_size
# sub_smpl = args.subsampling_parameter
# graph_saving_path = "./models/%s" % model_name
# ckpt_path = "%s/model.ckpt" % graph_saving_path


voc = pickle.load(open(args.voc_path, "rb"))
voc.prune(args.vocabulary_size)
voc.set_subsampling_param(args.subsampling_parameter)


reader = Reader(args.data_path,
                voc,
                args.context,
                args.window_size,
                args.negative,
                wiki=args.wiki,
                lang=args.language)


def next_batch(from_top_n=None):
    return reader.next_batch(from_top_n=from_top_n)

args.vocabulary_size = len(voc)
arg_dict = vars(args)
print(len(arg_dict))
for key, val in arg_dict.items():
    print("{}={}".format(key, val))


# def seld_line(a, p, l):
#     print("%d\t%d\t%d" % (a, p, l))
#     # if model_name != 'skipgram':
#     #     # t1 = " ".join(map(str,a))
#     #     # t2 = " ".join(map(str,p))
#     #     # t1 = " ".join([str(el) for el in a])
#     #     # t2 = " ".join([str(el) for el in p])
#     #     print("%s\t%s\t%d" % (a, p, l))
#     #     # sys.stdout.write("%s\t%s\t%d\n" % (a, p, l))
#     #
#     # else:
#     #     print("%d\t%d\t%d" % (a, p, l))
#     #     # sys.stdout.write("%s\t%s\t%d\n" % (a, p, l))


for e in range(args.epochs):

    print("epoch=%d" % e)
    batch = next_batch(from_top_n=args.vocabulary_size)

    while batch is not None:
        print(pickle.dumps(batch, protocol=4))
        for l in batch.tolist(): print(l)
        # for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
        #     seld_line(a, p, l)
        #     # print(voc.id2word[a], voc.id2word[p], l)
        #     pass
        batch = next_batch(from_top_n=args.vocabulary_size)
        # print("boom")
        sys.exit()

# epochs += 0