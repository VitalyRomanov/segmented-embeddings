from utils import WordSegmenter
import numpy as np

class MetaSegmenter:
    def __init__(self, paths, segm_lens, total_words):
        self.segmenter_gram = WordSegmenter(paths[0], segm_lens[0], total_words)
        self.segmenter_morph = WordSegmenter(paths[1], segm_lens[1], total_words)

        gram_offset = total_words
        morph_offset = gram_offset + self.segmenter_gram.unique_segments
        self.segm_voc_size = morph_offset + self.segmenter_morph.unique_segments + 1
        # add 1 for padding

        w2s = dict()
        for w in range(total_words):
            w2s[w] = [s_id + gram_offset for s_id in self.segmenter_gram.w2s[w]] + \
                     [s_id + morph_offset for s_id in self.segmenter_morph.w2s[w]]

        self.max_len = sum(segm_lens) + 1
        self.padding = self.segm_voc_size - 1

        segment_projection = []
        for w in w2s:
            # create padded blank
            z = np.ones((self.max_len,), dtype=np.int32) * self.padding
            # truncate previous segmentation to fit into a blank
            truncated = np.array(w2s[w][:min(self.max_len - 1, len(w2s[w]))],
                                 dtype=np.int32)
            # add word to the list of segments
            z[1:truncated.size + 1] = truncated
            z[0] = w
            segment_projection.append(z.reshape((1, -1)))

        self.segment_projection = np.vstack(segment_projection)

    def segment(self, batch):
        return self.segment_projection[batch, :]