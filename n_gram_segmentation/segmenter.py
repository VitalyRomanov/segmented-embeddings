import sys
import pickle

file_path = sys.argv[1]
lang = sys.argv[2]

words = open(file_path, "r").read().strip().split("\n")

char_gram_size = 3

char_grams = set()

def get_grams(w):
    if w[0] == '<':
        grams = [w]
    else:
        w = '<' + word + '>'
        grams = [w[i: i + char_gram_size] for i in range(len(w) - char_gram_size + 1)]
    return grams

with open("{}_word_{}_grams.txt".format(lang, char_gram_size), "w") as word_grams:
    for word in words:
        word_grams.write(word)
        word_grams.write("\t")

        grams = get_grams(word)
        for g in grams:
            word_grams.write(g)
            word_grams.write(" ")

            char_grams.add(g)
        
        word_grams.write("\n")
    
grams = list(char_grams)
grams.sort()

grams_dict = {}
for id_, g in enumerate(grams):
    grams_dict[g] = id_

print(len(grams))

word2gram = {}
for id_, word in enumerate(words):
    word2gram[id_] = [grams_dict[g] for g in get_grams(word)]

pickle.dump(word2gram, open("%s_word2segment.pkl" % lang, "wb"))
pickle.dump(grams_dict, open("%s_segment2id.pkl" % lang , "wb"))