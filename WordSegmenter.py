import pickle
import numpy as np

class WordSegmenter:
    def __init__(self, segmenter_files_path, lang, maximum_len=None):
        segment2id_path = "%s/%s_segment2id.pkl" % (segmenter_files_path, lang)
        word2segment_path = "%s/%s_word2segment.pkl" % (segmenter_files_path, lang)

        self.s2id = pickle.load(open(segment2id_path, "rb"))
        self.w2s = pickle.load(open(word2segment_path, "rb"))

        self.s2id["#"] = len(self.s2id)
        self.padding = self.s2id["#"]

        s, s_id = zip(*(self.s2id.items()))
        self.id2s = dict(zip(s_id, s))

        self.unique_segments = len(self.s2id)

        if maximum_len is None:
            self.max_len = 0
            for segm in self.w2s.values():
                if len(segm) > self.max_len:
                    self.max_len = len(segm)
        else:
            self.max_len = maximum_len

        self.max_len += 1

        total_words = len(self.w2s)
        self.padding += total_words

        for w in self.w2s:
            z = np.ones((self.max_len,), dtype=np.int32) * self.padding
            truncated = np.array(self.w2s[w][:min(self.max_len-1, len(self.w2s[w]))], dtype=np.int32) + total_words
            truncated = np.append(truncated, w)
            z[:truncated.size] = truncated
            self.w2s[w] = z

        self.w2s_str = {w: " ".join([str(el) for el in segm]) for w, segm in self.w2s.items()}


    def segment(self, batch, str_type=False):
        if str_type:
            return np.stack([self.w2s_str[id_] for id_ in batch])
        else:
            return np.stack([self.w2s[id_] for id_ in batch])

    def to_segments(self, batch):
        read = np.vectorize(lambda x: self.id2s[x])
        try:
            t = read(batch)
        except:
            print(batch)
            raise Exception()
        return t