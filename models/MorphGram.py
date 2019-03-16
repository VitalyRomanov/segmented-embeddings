from models import Fasttext
from utils import MetaSegmenter
import tensorflow as tf


class MorphGram(Fasttext):
    def __init__(self, vocab_size=None,
                 emb_size=None,
                 segmenter_path=None,
                 max_segments=None,
                 graph_path=None,
                 gpu_options=None,
                 ckpt_path=None,
                 model_name='morphgram'):
        self.graph_path = graph_path
        self.vocabulary_size = vocab_size
        self.emb_size = emb_size
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.segmenter = MetaSegmenter(segmenter_path, max_segments, vocab_size)

        self.max_segments = self.segmenter.max_len
        self.segm_voc_size = self.segmenter.segm_voc_size - 1

        self.assemble_model()

        self.open_session(gpu_options)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.graph_path,
                                                    graph=self.sess.graph)



