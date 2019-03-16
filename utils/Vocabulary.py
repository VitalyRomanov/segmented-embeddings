from collections import Counter
from scipy.stats import reciprocal
import numpy as np
import pickle


class Vocabulary:
    """
    Vocabulary stores the mapping from words to IDs in word2id,
    the inverse mapping in id2word
    frequency counts for words in id_count

    The class has method for sampling from the vocabulary according to some distribution

    """
    # indexes to keep word count
    word2id = None
    id2word = None
    id_count = None

    # arrays to keep probabilities for negative sampling and subsampling
    unigram_weights = None
    noise_weight = None  # unigram to power 3/4
    discard_prob = None  # discard probability according to word2vec papet
    subsampling_threshold = 1e-3  # used for calculating discard probability

    # handle OOV tokens
    unk_id = -1

    # auxiliary variables
    _total_count = 0

    def __init__(self, subsampling_threshold=1e-5):
        # initialize indexes
        self.word2id = {}
        self.word_count = Counter()  # used temporary to simplify the process of counting
        self.id_count = Counter()
        self.id2word = {}

        # buffers for random sampling
        self.negative_buffer = []
        self.uniform_buffer = np.array([])
        self.negative_buffer_position = 0
        self.uniform_buffer_position = 0
        self.subsampling_threshold = subsampling_threshold

        # flag to update weh nnew words are added
        self.to_update = True  # set when need to recaclulate indexes

    def _update(self):
        """
        Called before executing operations with vocabulary. Make sure all variables are up to date
        :return:
        """

        if self.to_update:
            words, word_ids = zip(*self.word2id.items())
            self.id2word = dict(zip(word_ids, words))

            counts = np.array([self.id_count[id_] for id_ in sorted(self.id_count.keys())])
            self.unigram_weights = counts / sum(counts)
            noise_weight = self.unigram_weights ** (3 / 4)
            # noise_weight = list(map(lambda x: reciprocal.pdf(x, 1, len(self)), range(1, len(self)+1)))
            self.noise_weight = noise_weight / sum(noise_weight)

            self.discard_prob = 1 - np.sqrt(self.subsampling_threshold / self.unigram_weights)

            self.negative_buffer = np.array([])
            self.to_update = False

            self._total_count = sum(self.id_count.values())

    def set_subsampling_param(self, param_val):
        self.subsampling_threshold = param_val
        self.to_update = True

    def subsample(self, token_ids):
        self._update()

        k = token_ids.size

        if k == 0:
            return np.array([])

        # calling np.random is slow. bufferize 100000 random samples and get slices every time the method is called
        if self.uniform_buffer_position + k > self.uniform_buffer.size:
            self.uniform_buffer = np.random.rand(max(1000000, k * 10))
            self.uniform_buffer_position = 0

        to_keep = np.greater(self.uniform_buffer[self.uniform_buffer_position: self.uniform_buffer_position + k],
                             self.discard_prob[token_ids])

        self.uniform_buffer_position += k

        return token_ids[to_keep]

    def get_id(self, word):
        """
        Return vocabulary ID of a word. Returns -1 if word is not in the vocabulary
        :param word:
        :return: None
        """
        return self.word2id.get(word, -1)

    def get_ids(self, words, subsample=True, select_top=None):
        ids = np.array([self.get_id(w) for w in words])

        if select_top is not None:
            ids[ids >= select_top] = self.unk_id

        ids[ids == -1] = self.unk_id
        ids = ids[ids != self.unk_id]

        if subsample:
            ids = self.subsample(ids)
        return ids

    def get_word(self, id_):
        return self.id2word[id_]

    def add_words(self, tokens):
        """
        Add new words to vocabulary. Updates only word_count
        :param tokens: list of string tokens
        :return: None
        """

        # u, counts = np.unique(np.array(tokens), return_counts=True)
        # for word, count in zip(u, counts):
        #     if word in self.word_count:
        #         self.word_count[word] += count
        #     else:
        #         self.word_count[word] = count
        for t in tokens:
            if t in self.word_count:
                self.word_count[t] += 1
            else:
                self.word_count[t] = 1

        self.to_update = True

    def prune(self, top_n):
        """
        Keep only top words in the vocabulary
        :param top_n:
        :return: None
        """

        total_before = sum(self.word_count.values())

        id_count = Counter()
        word2id = {}
        for ind, (word, count) in enumerate(self.word_count.most_common(top_n)):
            word2id[word] = ind
            id_count[ind] = count

        total_after = sum(id_count.values())
        word2id['<unk>'] = len(word2id)
        id_count[word2id['<unk>']] = total_before - total_after + 1

        self.word2id = word2id
        self.id_count = id_count
        self.unk_id = word2id['<unk>']

        self._update()

    def export_vocabulary(self, filename, top_n=-1):
        """
        Save words and their counts in TSV file
        :param top_n: how many words to export
        :param filename: where to export
        :return: None
        """

        self._update()

        if top_n == -1:
            top_n = len(self)

        with open(filename, "w") as voc_exp:
            voc_exp.write("{}\t{}\n".format("Word", "Count"))
            for word_id, count in self.id_count.most_common(top_n):
                voc_exp.write("{}\t{}\n".format(self.id2word[word_id], count / self.total_tokens()))

    def save(self, path):
        """
        Save vocabulary on disk
        :param path:
        :return: None
        """
        pickle.dump(self, open(path, "wb"))

    def save_wordcount(self, path):
        pickle.dump(self.word_count, open(path, "wb"))

    @classmethod
    def load(cls, path):
        """
        Load from disk
        :return: None
        """
        return pickle.load(open(path, "rb"))

    @classmethod
    def load_from_wordcount(cls, path):
        wc = pickle.load(open(path, "rb"))
        voc = Vocabulary()
        voc.word_count = wc
        return voc

    def total_tokens(self):
        """
        Return total number of tokens observed
        :return:
        """
        self._update()
        return self._total_count

    def __len__(self):
        return len(self.word2id)

    def in_top(self, word_id, top_n):
        return word_id >= top_n
        # if top_n not in self.top_sets:
        #     self.top_sets[top_n] = set(self.id_count.most_common(top_n).keys())
        #
        # return word_id in self.top_sets[top_n]

    def sample_negative(self, k):
        """
        Sample words according to noise distribution.
        :param k: Number of samples
        :return: sample as list
        """
        self._update()

        # calling np.random is slow. bufferize k*10 random samples and get slices every time the method is called
        if self.negative_buffer_position + k >= len(self.negative_buffer):
            self.negative_buffer = np.random.choice(np.array(list(self.id_count.keys())), 10000000, p=self.noise_weight)
            self.negative_buffer_position = 0

        sample = self.negative_buffer[self.negative_buffer_position: self.negative_buffer_position + k]
        self.negative_buffer_position += k

        return sample


if __name__ == "__main__":
    import sys
    from nltk import word_tokenize

    corpus_path = sys.argv[1]
    output_path = sys.argv[2]

    voc = Vocabulary()

    counter = 0

    with open(corpus_path, "r") as reader:
        line = reader.readline()
        while line:
            tokens = word_tokenize(line.strip(), preserve_line=True)
            voc.add_words(tokens)
            line = reader.readline()
            counter += 1
            if counter % 10000 == 0:
                print(counter)

    voc.prune(200000)
    print("Total {} tokens, unique {}".format(voc.total_tokens(), len(voc)))

    voc.export_vocabulary("voc.tsv")
    voc.save("voc.pkl")

    counter = 0

    with open(corpus_path, "r") as reader:
        with open(output_path, "w") as writer:
            line = reader.readline()
            while line:
                tokens = word_tokenize(line.strip(), preserve_line=True)

                if len(tokens) > 0:
                    new_tokens = voc.get_ids(tokens, select_top=1000)

                    for token_id in new_tokens:
                        writer.write("{} ".format(voc.get_word(token_id)))
                    writer.write("\n")

                line = reader.readline()
                counter += 1
                if counter % 10000 == 0:
                    print(counter)

            # print("Total {} tokens, unique {}".format(new_voc.total_tokens(), len(new_voc)))
            # new_voc.save("voc.pkl")
