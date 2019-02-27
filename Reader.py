import numpy as np
from Tokenizer import Tokenizer

class Reader:
    """
    Reader for implementing skipgram negative sampling
    Class exposes method next_minibatch

    takes a plain text file as input
    """
    def __init__(self, path, vocabulary, n_contexts, window_size, k, wiki=False, lang='en'):
        """

        :param path: Location of training file
        :param vocabulary: Vocabulary that can generate negative samples
        :param n_contexts: coxtexts to include in minibatch
        :param window_size: window size per context. full window span is window_size*2 + 1
        :param k: number of negative samples per context
        """
        self.wiki = wiki
        if wiki:
            
            if lang == 'en':
                from WikiLoaderJSON import WikiDataLoader
            elif lang == 'ru':
                from WikiLoaderv2 import WikiDataLoader
            self.readerClass = WikiDataLoader

        self.reader = None

        self.tokenizer = Tokenizer()

        # self.file = None
        self.path = path
        self.voc = vocabulary
        self.tokens = []

        self.window_size = window_size
        self.n_contexts = n_contexts
        self.k = k

        self.position = window_size

        self.init()

    def read_next(self):
        if self.wiki:
            return self.reader.next_doc()
        else:
            return self.reader.readline()

    def init(self):
        # If file is opened
        if self.wiki:
            if self.reader is not None:
                del self.reader
            self.reader = self.readerClass(self.path)
        else:
            if self.reader is not None:
                self.reader.close()
            self.reader = open(self.path, "r")
        
        # read initial set of tokens
        self.tokens = self.voc.get_ids(
            self.tokenizer(self.read_next().strip(), lower=True, hyphen=False)
            ).tolist()

    def get_tokens(self, from_top_n):
        nl = self.read_next()
        if nl == "" or nl is None:
            self.tokens = None    
        else:
            # discard old tokens and append new ones
            new_tokens = self.tokenizer(self.read_next().strip(), lower=True, hyphen=False)
            token_ids = self.voc.get_ids(new_tokens, select_top=from_top_n)
            self.tokens = self.tokens[self.position - self.window_size: -1] + token_ids.tolist()
            self.position = self.window_size

    def next_batch(self, from_top_n=None):
        """
        Generate next minibatch. Only words from vocabulary are included in minibatch
        :return: batches for (context_word, second_word, label)
        """

        batch = []
        context_count = 0

        while self.tokens is not None and context_count < self.n_contexts:
            # get more tokens if necessary
            while self.tokens is not None and self.position + self.window_size + 1 > len(self.tokens):
                self.get_tokens(from_top_n)

            # re-initialize if at the end of the file
            if self.tokens is None:
                self.init()
                return None

            c_token_id = self.tokens[self.position]


            # generate positive samples
            for i in range(-self.window_size, self.window_size + 1, 1):
                if i == 0:
                    continue

                c_pair_id = self.tokens[self.position + i]

                batch.append([c_token_id, c_pair_id, 1.])

            # generate negative samples
            neg = self.voc.sample_negative(self.k)
            for n in neg:
                # if word is the same as central word, the pair is ommited
                if n != c_token_id:
                    batch.append([c_token_id, n, 0.])

                context_count += 1

            self.position += 1

        bb = np.array(batch)
        return bb[:,0], bb[:,1], bb[:,2]