import pickle
import numpy as np


class WordSegmenter:
    def __init__(self,
                 segmenter_files_path,
                 maximum_segments=None,
                 total_words=-1,
                 include_word=True):

        segment2id_path = "%s/segment2id.pkl" % (segmenter_files_path)
        word2segment_path = "%s/word2segment.pkl" % (segmenter_files_path)

        self.id2s = inverse_dict(pickle.load(open(segment2id_path, "rb")))
        self.w2s = pickle.load(open(word2segment_path, "rb"))

        if total_words == -1:
            self.total_words = len(self.w2s)
        else:
            self.total_words = total_words
            self.prune_segments(total_words)

        self.padding = len(self.id2s)
        self.id2s[self.padding] = '#'
        self.padding += (self.total_words if include_word else 0)  # offset to include words as segments

        self.unique_segments = len(self.id2s)

        add_one = 1 if include_word else 0

        # add 1 to include word as a segment
        if maximum_segments is None:
            self.max_len = max(map(len, self.w2s.values())) + add_one
        else:
            self.max_len = maximum_segments + add_one

        self.w2s_lens = clipped_lengths(self.w2s, self.max_len)
        self.len_proj = len_projector(self.w2s, self.max_len)

        segment_projection = []
        for w in self.w2s:
            # create padded blank
            z = np.ones((self.max_len,), dtype=np.int32) * self.padding
            # truncate previous segmentation to fit into a blank
            # offset all segment ids by vocabulary size
            truncated = np.array(self.w2s[w][:min(self.max_len - add_one, len(self.w2s[w]))],
                                 dtype=np.int32) + (self.total_words if include_word else 0)
            # add word to the list of segments
            z[add_one:truncated.size + add_one] = truncated
            if include_word:
                z[0] = w

            segment_projection.append(z.reshape((1,-1)))

        self.segment_projection = np.vstack(segment_projection)

        # self.w2s_str = {w: " ".join([str(el) for el in segm]) for w, segm in self.w2s.items()}

    def get_lens(self, items):
        # return np.array([self.w2s_lens[i] for i in items], dtype=np.int32)
        return self.len_proj[items]

    def segment(self, batch):
    # def segment(self, batch, str_type=False):
        # if str_type:
        #     return np.stack([self.w2s_str[id_] for id_ in batch])
        # else:
        #     return np.stack([self.w2s[id_] for id_ in batch])
        return self.segment_projection[batch,:]


    def from_segments(self, batch):
        read = np.vectorize(lambda x: self.id2s[x])
        try:
            t = read(batch - self.total_words)
        except:
            print(batch)
            raise Exception()
        return t

    def prune_segments(self, top_n_words):
        segments_to_keep = set()

        for w in self.w2s:
            if w < top_n_words:
                for s in self.w2s[w]:
                    segments_to_keep.add(s)

        for w in range(top_n_words, len(self.w2s)):
            self.w2s.pop(w)

        for s in range(len(self.id2s)):
            if s not in segments_to_keep:
                self.id2s.pop(s)

        id2id = dict(zip(self.id2s.keys(), range(len(self.id2s))))
        id2s = {id2id[id_]: self.id2s[id_] for id_ in self.id2s}
        w2s = dict()
        for w in self.w2s:
            w2s[w] = [id2id[id_] for id_ in self.w2s[w]]

        self.id2s = id2s
        self.w2s = w2s

        # print("finished prunning")


def inverse_dict(d):
    key, vals = zip(*(d.items()))
    return dict(zip(vals, key))


def clipped_lengths(segmentation, max_len):
    w2s_lens = dict()
    for w in segmentation:
        w2s_lens[w] = min(max_len - 1, len(segmentation[w])) + 1
    return w2s_lens

def len_projector(segmentation, max_len):
    return np.array([min(max_len - 1, len(segmentation[w])) + 1 for w in range(len(segmentation))], dtype=np.int32)
