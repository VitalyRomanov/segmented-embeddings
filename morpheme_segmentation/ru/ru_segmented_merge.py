import pickle
import sys

original_path = sys.argv[1]
segmented_path = sys.argv[2]

segmented = open(segmented_path, "r").read().strip().split("\n")
original = open(original_path, "r").read().strip().split("\n")

word2morph = {}
morphemes = set()
with open("ru_segmented_correct.txt", "w") as ru_segm:
    for i in range(len(original)):
        s = segmented[i].split("\t")[0]
        if s != original[i]:
            # print(s, original[i])

            m = "<" + original[i] + ">"

            ru_segm.write("{}\t{}\n".format(original[i], m))            

            morphemes.add(m)
            word2morph[original[i]] = [m]
        else:
            # ru_segm.write("{}\n".format(segmented[i]))
            
            ms = segmented[i].split("\t")[1].strip().split()
            ms[0] = "<" + ms[0]
            ms[-1] = ms[-1] + ">"

            ru_segm.write(original[i])
            ru_segm.write("\t")
            for m in ms: ru_segm.write("%s " % m)
            ru_segm.write("\n")

            for m in ms:
                morphemes.add(m)

            word2morph[original[i]] = ms


morphemes = list(morphemes)
morphemes.sort()

print(len(morphemes))

m2id = dict(zip(morphemes, range(len(morphemes))))

w2m = {}

for id_, word in enumerate(original):
    w2m[id_] = [m2id[m] for m in word2morph[word]]
    # try:
    #     
    # except:
    #     print(word2morph[word], word2morph[word])
    #     raise Exception()

pickle.dump(w2m, open("ru_word2segment.pkl", "wb"))
pickle.dump(m2id, open("ru_segment2id.pkl", "wb"))
            
            