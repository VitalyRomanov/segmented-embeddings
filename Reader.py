import numpy as np
from Tokenizer import Tokenizer

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

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
                from WikiLoaderv2 import WikiDataLoader
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
            )

    def get_tokens(self, from_top_n):
        nl = self.read_next()
        if nl == "" or nl is None:
            self.tokens = None
        else:
            # discard old tokens and append new ones
            new_tokens = self.tokenizer(nl.strip(), lower=True, hyphen=False)
            token_ids = self.voc.get_ids(new_tokens, select_top=from_top_n)
            # self.tokens = self.tokens[self.position - self.window_size: -1] + token_ids.tolist()
            self.tokens = np.concatenate([self.tokens[self.position - self.window_size: -1], token_ids], axis=0)
            self.position = self.window_size

    def next_batch(self, from_top_n=None, segmenter=None):
        """
        Generate next minibatch. Only words from vocabulary are included in minibatch
        :return: batches for (context_word, second_word, label)
        """

        batch = []
        context_count = 0

        while self.tokens is not None and self.position + self.n_contexts + 2*self.window_size + 1 > len(self.tokens):
            self.get_tokens(from_top_n)

        if self.tokens is None:
            self.init()
            return None

        tokens = self.tokens[self.position: self.position + self.n_contexts + 2*self.window_size]

        windows = rolling_window(tokens, self.window_size * 2 + 1)

        neg = self.voc.sample_negative(self.k * self.n_contexts)

        if segmenter is None:

            context_words = np.delete(windows, self.window_size, axis=1).T.reshape((-1,))

            central_ = np.tile(windows[:, self.window_size], (2 * self.window_size + self.k,))
            context_ = np.concatenate([context_words, neg], axis=0)
            labels_ = np.concatenate([np.ones_like(context_words), np.zeros_like(neg)], axis=0)

            batch_ = np.hstack([central_[:, None], context_[:, None], labels_[:, None]])

        else:

            segments = segmenter(windows[:, self.window_size])
            context_words = np.delete(windows, self.window_size, axis=1)
            context_negative = np.hstack([context_words, neg.reshape(self.n_contexts, self.k)])
            contexts = []
            for ind, segment in enumerate(segments):
                c_n = context_negative[ind, :]
                segment_context_pairs = np.transpose([np.tile(segment, len(c_n)), np.repeat(c_n, len(segment))])
                lbls = np.ones((segment.size * c_n.size, 1), dtype=np.int32)
                lbls[-self.k*segment.size:] = 0
                segment_batch = np.hstack([segment_context_pairs, lbls])
                contexts.append(segment_batch)

            batch_ = np.vstack(contexts)

        self.position += self.n_contexts + self.window_size

        return batch_