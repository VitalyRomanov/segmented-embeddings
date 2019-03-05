import sys
import numpy as np

from collections import Counter
path = sys.argv[1]

with open(path, "r") as file_:
    lines = file_.read().strip().split("\n")
    seg = list(map(lambda x:len(x.split()), map(lambda x: x.split("\t")[1], lines)))
    print(np.mean(seg))
    print(np.std(seg))
    print(np.max(seg))

    for c, v in Counter(seg).most_common():
        print(v, c)