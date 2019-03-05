import pickle
import sys
from WordSegmenter import WordSegmenter

lang = sys.argv[1]
voc_p = sys.argv[2]
segments_p = sys.argv[3]
att_p = sys.argv[4]

attention = pickle.load(open(att_p, "rb"))
ws = WordSegmenter(segments_p, lang)

n_words = attention.shape[0]

with open(voc_p, "r") as v_file:
    lines = v_file.read().strip().split()[:n_words]
    words = map(lambda x: x.split()[0], lines)
    voc = dict(zip(range(n_words), words))

print("Ready")
while 1:
    ind = int(input().strip())
    segments = ws.segment([ind]).reshape((-1,))
    sgm = [voc[segments[0]]] + ws.to_segments(segments[1:]).tolist()
    att = attention[ind].reshape((-1,)).tolist()
    print(len(sgm))
    print(len(att))
    for p, a in zip(sgm, att):
        print("%s\t%.4f" % (p, a))