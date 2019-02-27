# from WikiLoaderJSON import WikiDataLoader
from Vocabulary import Vocabulary
import sys
from collections import Counter
from nltk import word_tokenize
from Tokenizer import Tokenizer

verb_dep_counter = Counter()

if __name__=="__main__":

    if len(sys.argv) != 4:
        raise Exception("Usage:\n\tpython create_voc.py lang path/to/wiki out/filename")

    lang = sys.argv[1]
    wiki_path = sys.argv[2]
    out_name = sys.argv[3]

    if lang == 'en':
        from WikiLoaderJSON import WikiDataLoader
    elif lang == 'ru':
        from WikiLoaderv2 import WikiDataLoader

    voc = Vocabulary()

    wiki = WikiDataLoader(wiki_path)
    wiki_doc = wiki.next_doc()
    tok = Tokenizer()

    counter = 0 
    
    while wiki_doc:
        # tokens = word_tokenize(wiki_doc.strip(), preserve_line=True)
        tokens = tok(wiki_doc, lower=True, hyphen=False)
        # print(tokens)
        # sys.exit()
        voc.add_words(tokens)
        counter += 1
        if counter % 10000 == 0:
            print(counter)

        wiki_doc = wiki.next_doc()

    voc.prune(200000)
    print("Total {} tokens, unique {}".format(voc.total_tokens(), len(voc)))

    voc.export_vocabulary("%s.tsv" % out_name)
    voc.save("%s.pkl" % out_name)
                
