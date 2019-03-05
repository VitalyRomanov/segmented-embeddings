import sys
from WikiLoaderv2 import WikiDataLoader

wiki_path = sys.argv[1]
out_name = sys.argv[2]

wiki = WikiDataLoader(wiki_path)
wiki_doc = wiki.next_doc()

with open(out_name, "w") as wd:
    while wiki_doc:
        wd.write("%s\n" % wiki_doc)
        wiki_doc = wiki.next_doc()