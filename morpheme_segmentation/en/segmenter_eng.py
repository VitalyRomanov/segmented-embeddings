# polyglot download morph2.en
from collections import Counter
from polyglot.text import Word
import pickle
import sys

file_path = sys.argv[1]

lines = open(file_path).read().strip().split("\n")

morphemes = set()

def to_morphemes(word):
    if word[0] == "<":
        ms = [word]
    else:
        ms = list(map(str, Word(word, language="en").morphemes))
    return ms

word2id = {}
# morph_count = Counter()

with open("en_segmentation.txt", "w") as en_segm:
    for id_, line in enumerate(lines):
        word = line.split("\t")[0]

        en_segm.write(word)
        en_segm.write("\t")

        ms = to_morphemes(word)
        ms[0] = "<" + ms[0]
        ms[-1] = ms[-1] + ">"

        for m in ms:
            morphemes.add(m)

            en_segm.write(m)
            en_segm.write(" ")

        en_segm.write("\n")

        word2id[word] = id_



morphemes = list(morphemes)
morphemes.sort()

print(len(morphemes))

morpheme_dict = {}

for id_, m in enumerate(morphemes):
    morpheme_dict[m] = id_

word2morph = {}

for word, id_ in word2id.items():
    ms = to_morphemes(word)
    ms[0] = "<" + ms[0]
    ms[-1] = ms[-1] + ">"
    word2morph[id_] = [morpheme_dict[m] for m in ms]

del Word

pickle.dump(word2morph, open("en_word2segment.pkl", "wb"))
pickle.dump(morpheme_dict, open("en_segment2id.pkl", "wb"))

# from pprint import pprint
# pprint(word2morph)