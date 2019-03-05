import pickle
import argparse

parser = argparse.ArgumentParser(description='Save embeddings into w2v format')
parser.add_argument('emb_path', type=str, default="", help='Path to pkl file with embeddings')
parser.add_argument('voc_path', type=str, default="", help='Path to TSV vocabulary')
parser.add_argument('out_path', type=str, default="", help='Where to store result')

args = parser.parse_args()


def parse_voc(file_path):
    with open(file_path, "r") as voc:
        lines = voc.read().strip().split("\n")
        words = list(map(lambda x: x.split()[0], lines))
        return words [1:] # remove header


def read_embs(emb_path):
    return pickle.load(open(emb_path, "rb"))


words = parse_voc(args.voc_path)
embs = read_embs(args.emb_path)


with open(args.out_path, "w") as w2v:
    voc_size = embs.shape[0]
    emb_dim = embs.shape[1]

    # write sizes
    w2v.write("%d %d\n" % (voc_size, emb_dim))

    for ind in range(voc_size):
        # emb to str
        emb_string = " ".join(list(map(str, embs[ind])))
        # write to file
        w2v.write("%s %s\n" % (words[ind], emb_string))