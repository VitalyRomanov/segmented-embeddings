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
            self.tokenizer(self.read_next().strip(), lower=True, hyphen=False), subsample=True
        )

    def get_tokens(self, from_top_n):
        nl = self.read_next()
        if nl == "" or nl is None:
            self.tokens = None
        else:
            # discard old tokens and append new ones
            new_tokens = self.tokenizer(nl.strip(), lower=True, hyphen=False)
            token_ids = self.voc.get_ids(new_tokens, select_top=from_top_n, subsample=True)
            # self.tokens = self.tokens[self.position - self.window_size: -1] + token_ids.tolist()
            self.tokens = np.concatenate([self.tokens[self.position - self.window_size: -1], token_ids], axis=0)
            self.position = self.window_size

    def next_batch(self, from_top_n=None):
        """
        Generate next minibatch. Only words from vocabulary are included in minibatch
        :return: batches for (context_word, second_word, label)
        """

        while self.tokens is not None and self.position + self.n_contexts + 2 * self.window_size + 1 > len(self.tokens):
           self.get_tokens(from_top_n)

        if self.tokens is None:
           self.init()
           return None

        w_span = 2 * self.window_size
        w_s = self.window_size
        n_c = self.n_contexts
        k = self.k

        # w_s -= np.random.randint(self.window_size - 1)

        tokens = self.tokens[self.position - w_s: self.position + n_c + w_s]
        self.position += n_c + w_s

        windows = rolling_window(tokens, w_span + 1)

        central_w = np.tile(windows[:, w_s].reshape((-1,1)), (1, w_span)).reshape((-1,))
        central_neg = np.tile(windows[:, w_s].reshape((-1,1)), (1, k)).reshape((-1,))
        central_ = np.concatenate([central_w, central_neg], axis=0)

        context_w = np.delete(windows, w_s, axis=1).reshape((-1,))
        context_neg = self.voc.sample_negative(k * n_c)
        context_ = np.concatenate([context_w, context_neg], axis=0)

        labels_ = np.concatenate([np.ones_like(context_w), np.zeros_like(context_neg)], axis=0)

        batch_ = np.hstack([central_[:, None], context_[:, None], labels_[:, None]])

        return batch_
        # return batch_[:, 0], batch_[:, 1], batch_[:, 2]


        # ##### Old Long Version
        # tokens = self.tokens[self.position: self.position + self.n_contexts + 2*self.window_size + 5]
        # # token shape (self.n_contexts + 2*self.window_size,)
        #
        # windows = rolling_window(tokens, self.window_size * 2 + 1)
        # # windows shape (self.n_contexts, 2*self.window_size + 1)
        #
        # neg = self.voc.sample_negative(self.k * self.n_contexts)
        # # neg shape (self.n_contexts * self.k, )
        #
        # central_w = (windows[:, self.window_size].reshape(-1,1) * np.ones((1, 2 * self.window_size), dtype=np.int32)).reshape((-1,))
        # # central_w shape (self.n_contexts, 2 * self.window_size)
        # central_neg = (windows[:, self.window_size].reshape(-1, 1) * np.ones((1, self.k), dtype=np.int32)).reshape((-1,))
        # # central_neg shape (self.n_contexts, self.k)
        #
        # context_words = np.concatenate([windows[:, :self.window_size], windows[:, self.window_size + 1:]], axis=1).reshape((-1,))
        #
        # central_ = np.concatenate([central_w, central_neg], axis=0)
        # context_ = np.concatenate([context_words, neg], axis=0)
        # labels_ = np.concatenate([np.ones_like(context_words), np.zeros_like(neg)], axis=0)
        # return batch_, context_, labels_
        ##

        #########################

        batch = []
        context_count = 0

        while self.tokens is not None and context_count < self.n_contexts:
            # get more tokens if necessary
            while self.tokens is not None and self.position + 2 * self.window_size + 1 > len(self.tokens):
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

        bb = np.array(batch, dtype=np.int32)
        return bb
        # return bb[:, 0], bb[:, 1], bb[:, 2]